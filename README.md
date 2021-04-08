# Tibetan Search Engine

TF-IDF based search engine for Tibetan documents


# Usage

```bash
# note: please ensure that the `MySQL` and `Redis` server are installed properly

pip install -r requirements.txt

# create database `corpus` and insert the documents under the folder `data/corpus/<text_type>/`

flask init-db

# create the TF-IDF vectorizer and extract the features

flask init-tfidf

# run the server

flask run
```
