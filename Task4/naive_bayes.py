#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/6/30 下午6:22
@Author : mia
"""
import os
import jieba
import pandas
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#preprocess用于将一个文本文档进行切词，并以字符串形式输出切词结果
def preprocess(path_name):
    text_with_spaces=""
    textfile=open(path_name,"r",encoding="gbk").read()
    textcut=jieba.cut(textfile)
    for word in textcut:
        text_with_spaces+=word+" "
    return text_with_spaces


#loadtrainset用于将某一文件夹下的所有文本文档批量切词后，载入为训练数据集；返回训练集和每一个文本（元组）对应的类标号。
def loadtrainset(path,classtag):
    allfiles=os.listdir(path)
    processed_textset=[]
    allclasstags=[]
    for thisfile in allfiles:
        path_name=path+"/"+thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    return processed_textset,allclasstags

def loadtrainset1(path):
    all_examples, labels = [], []
    with open(path,'r') as f:
        content = f.readlines()
        for line in content:
            all_examples.append(line.split('\t')[0])
            labels.append(line.split('\t')[1])
    return all_examples, labels


# processed_textdata1,class1=loadtrainset("F:/Datasets/玄幻", "玄幻")
# processed_textdata2,class2=loadtrainset("F:/Datasets/科幻", "科幻")
# processed_textdata3,class3=loadtrainset("F:/Datasets/都市", "都市")
# integrated_train_data=processed_textdata1+processed_textdata2+processed_textdata3
# classtags_list=class1+class2+class3
integrated_train_data,classtags_list = loadtrainset1('../data/cnews/cnews.train.txt')


count_vector = CountVectorizer()
#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vector_matrix = count_vector.fit_transform(integrated_train_data)

#tfidf度量模型
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
#将词频矩阵转化为权重矩阵,每一个特征值就是一个单词的TF-IDF值


#调用MultinomialNB分类器进行训练
clf = MultinomialNB().fit(train_tfidf,classtags_list)#


#测试
testset=[]
testset.append(preprocess("/Users/mia/Desktop/mia/NLP_Learning/data/cnews/one_test.txt"))
new_count_vector = count_vector.transform(testset)
new_tfidf= TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
predict_result = clf.predict(new_tfidf) #预测结果
print(predict_result)