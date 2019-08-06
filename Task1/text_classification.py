#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/8/6 下午9:54
@Author : mia
"""

from collections import Counter
import numpy as np


def load_data(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass

    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表"""
    data_train, _ = load_data(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1) # 找到词频最大的5000-1个字
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    with open(vocab_dir, 'w') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(train_dir):
    _, categories = load_data(train_dir)
    categories = list(set(categories))
    cate_to_id = dict(zip(categories, range(len(categories))))
    print(cate_to_id)
    return categories, cate_to_id




if __name__ == '__main__':
    # build_vocab('../data/cnews/cnews.train.txt', vocab_dir='../data/cnews/vacab.txt')
    # words, word_to_id = read_vocab('../data/cnews/vacab.txt')
    # print(words)
    # print(word_to_id)
    read_category('../data/cnews/cnews.train.txt')