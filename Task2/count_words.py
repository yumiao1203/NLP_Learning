#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/6/23 下午11:41
@Author : mia
"""
from collections import Counter
import jieba


def isChinese(str):
    if str >= '\u4e00' and str<= '\u9fa5':
        return True
    else:
        return False

def get_text_CN(text_path):
    with open(text_path,'r',encoding='utf8') as f:
        content_list= f.readlines()
    return content_list

content_list = get_text_CN('/Users/mia/Desktop/mia/NLP_Learning/data/cnews/cnews.test.txt')
test_line =content_list[0]
print(test_line)
test_line = test_line.strip(' ')
words_list = list(jieba.cut(test_line,cut_all=False))
print(words_list)
for ws in words_list:
    print(ws)
count_zi = Counter(test_line) # 计算单字的频次
count_ci = Counter(words_list) # 计算词语的频次
print(count_zi)
print(count_ci)




