#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/7/6 下午7:58
@Author : mia
"""

import logging
import fasttext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

classifier = fasttext.supervised('data/cnews/cnews.train.txt',
                                 'data/cnews/news_fasttext.model',
                                 label_prefix='__label__')
classifier = fasttext.load_model('data/cnews/news_fasttext.model.bin',
                                 label_prefix='__label__')
result = classifier.test('data/cnews/news.test.txt')
print(result.precision)
print(result.recall)