import glob

import os

import pymysql

import redis

import click

from flask import Flask, request, g, render_template

from tqdm import tqdm

from search import SearchEngine, tokenize


app = Flask(__name__)

app.config.update(
    # where corpus are stored and retrieved

    DATABASE='corpus',
    DATABASE_USER='test',
    DATABASE_PASSWD='123456',

    # cache search result in redis

    REDIS_DATABASE='redis://',

    # cache search result for 10 minutes

    CACHE_RESULT_TIME=10 * 60,

    # ignore texts whose similarity score less than 0.05

    MIN_SCORE=0.05,

    # number of search result per page

    PAGE_SIZE=10,

    # number of available page links

    PAGE_LINK=10,
)


def get_db():
    if 'db' not in g:
        db = pymysql.connect(
            user=app.config['DATABASE_USER'],
            passwd=app.config['DATABASE_PASSWD'],
        )

        db_name = app.config['DATABASE']

        with db.cursor() as cursor:
            if cursor.execute('SHOW DATABASES LIKE %s', (db_name,)):
                db.select_db(db_name)

        g.db = db

    return g.db


@app.teardown_appcontext
def close_db(e=None):
    if 'db' in g:
        g.pop('db').close()


def _init_db():
    with open('schema.sql', encoding='utf-8') as f:
        stmts = f.read().split(';\n')

    corpus = []

    for file in glob.glob('data/corpus/*/*.txt'):
        text_type = os.path.basename(
            os.path.dirname(file)
        )

        # remove BOM if present

        with open(file, encoding='utf-8-sig') as f:
            text = f.read()

        # remove redundant blanks

        text = ' '.join(text.split())

        # remove duplicates

        row = (text_type, text)

        if row not in corpus:
            corpus.append(row)

    db = get_db()

    with db.cursor() as cursor:
        for stmt in stmts:
            cursor.execute(stmt)

        cursor.executemany('INSERT INTO corpus (type, body) VALUES (%s, %s)', corpus)

    db.commit()


@app.cli.command('init-db')
def init_db():
    _init_db()

    click.echo('Initialized the database.\n'
               "Don't forget to run `init-tfidf` command if you've modified corpus.")


def get_rdb():
    if 'rdb' not in g:
        g.rdb = redis.from_url(
            app.config['REDIS_DATABASE'], decode_responses=True
        )

    return g.rdb


def _init_tfidf():
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute('SELECT body FROM corpus')

        rows = cursor.fetchall()

    return SearchEngine(tqdm([
        r[0] for r in rows
    ]))


@app.cli.command('init-tfidf')
def init_tfidf():
    _init_tfidf()

    click.echo('Initialized the TF-IDF.')


def cache_search_result(keyword, text_ids):
    text_ids = dict(text_ids)

    r = get_rdb()

    total_num = r.zadd(keyword, text_ids)

    r.expire(keyword, app.config['CACHE_RESULT_TIME'])

    return total_num


def restore_search_result(keyword, start=0, end=-1, return_total_num=False):
    r = get_rdb()

    text_ids = r.zrange(
        keyword,
        start,
        end, # inclusive
        desc=True,
        withscores=True,
    )

    # extend life

    r.expire(keyword, app.config['CACHE_RESULT_TIME'])

    if return_total_num:
        return text_ids, r.zcard(keyword)
    else:
        return text_ids


try:
    search_engine = SearchEngine()
except OSError:
    pass


@app.route('/search')
def index():
    keyword = request.args.get('keyword', '')
    page = request.args.get('page', 1, int) - 1

    size = app.config['PAGE_SIZE']

    texts = []
    pages = []

    if keyword:
        start = page * size
        end = start + size

        text_ids, total_num = restore_search_result(keyword, start, end - 1, return_total_num=True)

        page_num = total_num / size

        if not text_ids and 0 <= page < page_num:
            # avoid concurrency error

            text_ids = restore_search_result(keyword, start, end - 1)

        elif not total_num:
            scores = search_engine(keyword)

            if scores.size:
                db = get_db()

                with db.cursor() as cursor:
                    cursor.execute('SELECT id FROM corpus')

                    rows = cursor.fetchall()

                text_ids = []

                for ind, score in scores:
                    ind = int(ind)

                    text_id = rows[ind][0]

                    text_ids.append((text_id, score))

                total_num = cache_search_result(keyword, text_ids)

                page_num = total_num / size

                # get the interesting part

                text_ids = text_ids[start:end]

        if text_ids:
            db = get_db()

            with db.cursor() as cursor:
                for text_id, score in text_ids:
                    cursor.execute('SELECT body FROM corpus WHERE id = %s', (text_id,))

                    row = cursor.fetchone()

                    text = row[0]

                    words = tokenize(text)

                    texts.append({
                        'id': text_id,
                        'body': text,
                        'size': len(words),
                        'score': score,
                    })

        link_num = app.config['PAGE_LINK']

        i = max(0, page - link_num // 2)

        page_end = min(page_num, i + link_num)

        while i < page_end:
            i += 1

            pages.append(i)

    return render_template('index.html', keyword=keyword, texts=texts, pages=pages)
