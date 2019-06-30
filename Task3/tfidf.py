#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/6/26 下午8:38
@Author : mia
"""
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I come to China to travel",
          "This is a car polupar in China",
          "I love tea and Apple ",
          "The work is to write some papers in science"]

vectorizer=CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print (tfidf)