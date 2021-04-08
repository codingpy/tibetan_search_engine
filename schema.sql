CREATE DATABASE IF NOT EXISTS corpus;

USE corpus;

DROP TABLE IF EXISTS corpus;

CREATE TABLE corpus (
  id INT PRIMARY KEY AUTO_INCREMENT,
  type TEXT NOT NULL,
  body LONGTEXT NOT NULL
);