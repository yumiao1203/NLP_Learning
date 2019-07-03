#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/7/3 下午8:04
@Author : mia
"""

from gensim.models import word2vec
import logging

##训练word2vec模型

# 获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 加载分词后的文本，使用的是Text8Corpus类

sentences = word2vec.Text8Corpus(r'/Users/mia/Desktop/mia/NLP_Learning/data/cnews/cnews.train.txt')

# 训练模型，部分参数如下
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)

# 模型的预测
print('-----------------分割线----------------------------')

# 计算两个词向量的相似度
try:
    sim1 = model.similarity(u'中央企业', u'事业单位')
    sim2 = model.similarity(u'教育网', u'新闻网')
except KeyError:
    sim1 = 0
    sim2 = 0
print(u'中央企业 和 事业单位 的相似度为 ', sim1)
print(u'人民教育网 和 新闻网 的相似度为 ', sim2)

print('-----------------分割线---------------------------')
# 与某个词（李达康）最相近的3个字的词
print(u'与国资委最相近的3个字的词')
req_count = 5
for key in model.similar_by_word(u'国资委', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

print('-----------------分割线---------------------------')
# 计算某个词(侯亮平)的相关列表
try:
    sim3 = model.most_similar(u'新华社', topn=20)
    print(u'和 新华社 与相关的词有：\n')
    for key in sim3:
        print(key[0], key[1])
except:
    print(' error')

print('-----------------分割线---------------------------')
# 找出不同类的词
sim4 = model.doesnt_match(u'新华社 人民教育出版社 人民邮电出版社 国务院'.split())
print(u'新华社 人民教育出版社 人民邮电出版社 国务院')
print(u'上述中不同类的名词', sim4)

print('-----------------分割线---------------------------')
# 保留模型，方便重用
model.save(u'搜狗新闻.model')

# 对应的加载方式
# model2 = word2vec.Word2Vec.load('搜狗新闻.model')
# 以一种c语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)
