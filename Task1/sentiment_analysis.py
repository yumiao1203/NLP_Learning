#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/6/21 上午11:33
@Author : mia
"""
import tensorflow as tf
import keras
# from tensorflow import keras
import glob
import codecs
import os
import numpy
import nltk
print(tf.__version__)

# 这个数据跑不下来
imdb = keras.datasets.imdb
print(imdb)
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/imdb.npz',num_words=15000)

# train_neg_path = glob.glob('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/neg/*.txt')
# train_pos_path = glob.glob('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/pos/*.txt')
# pos_all = codecs.open('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/pos_all.txt', 'a', encoding='utf8')
# neg_all = codecs.open('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/neg_all.txt', 'a', encoding='utf8')
#
#
# #将单独的txt合并成一个pos_all.txt neg_all.txt
# def combine_train_data(train_pos_path, train_neg_path):
#     pos_contents = []
#     for train_pos_file in train_pos_path:
#         with open(train_pos_file,'r',encoding='utf8') as f:
#             content = f.readlines()
#             pos_contents.extend(content)
#
#     for pos_content in pos_contents:
#         pos_all.write(pos_content)
#         pos_all.write('\n')
#     pos_all.close()
#
#     neg_contents = []
#     for train_neg_file in train_neg_path:
#         with open(train_neg_file,'r',encoding='utf8') as f:
#             content = f.readlines()
#             neg_contents.extend(content)
#
#     for neg_content in neg_contents:
#         neg_all.write(neg_content)
#         neg_all.write('\n')
#     neg_all.close()
#     print(len(pos_contents))
#     print(len(neg_contents))
#
#
# def load_train_data():
#     pos_list = []
#     with open('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/pos_all.txt','r',encoding='utf8') as f:
#         line = f.readlines()
#         pos_list.extend(line)
#     neg_list = []
#     with open('/Users/mia/Desktop/mia/NLP_Learning/Task1/data/imdb/train/neg_all.txt','r',encoding='utf8') as f:
#         line = f.readlines()
#         neg_list.extend(line)
#     # 创建标签
#     label = [1 for i in range(len(pos_list))]
#     label.extend([0 for i in range(len(neg_list))])
#     # 评论内容整合
#     pos_list.extend(neg_list)
#     content = pos_list
#     print(len(label))
#     print(len(content))
#
#
# 预处理，去停用词和去标点符号
def preprocess():
    seq = []
    seqtence = []
    stop_words = set(stopwords.words('english'))
    for con in content:
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        seq.append(line)
        seqtence.extend(line)

#
# combine_train_data(train_pos_path, train_neg_path)
# load_train_data()

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#查看第一个和第二个评论的字数
print(len(train_data[0]))
print(len(train_data[1]))

# Convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[20])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)







