#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:34:20 2018

@author: liuxueying
"""


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.externals import joblib
from time import time


#tfid
def feature_extractor(input_x):
    return TfidfVectorizer().fit_transform(input_x)

#MultinomialNB
def fit_and_predicted(train_x, train_y, test_x, test_y):
    """
    训练与预测
    :param train_x: 
    :param train_y: 
    :param test_x: 
    :param test_y: 
    :return: 
    """
    clf = MultinomialNB().fit(train_x, train_y)
    joblib.dump(clf, 'model.pkl')
    predicted = clf.predict(test_x)
    print(metrics.classification_report(test_y, predicted))
    print('accuracy_score: %0.5fs' %(metrics.accuracy_score(test_y, predicted)))


#读取数据

clean_train_data = []
train_label_list = []
clean_test_data = []
test_label_list = []

with open('cnews/cnews.train.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    train_label_list = [k.split()[0] for k in line_list]
    


with open('cnews/cnews.test.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    test_label_list = [k.split()[0] for k in line_list]


with open('cnews/data/clean_train_data.txt', 'r') as f3:
    clean_train_data = f3.readlines()

with open('cnews/data/clean_test_data.txt', 'r') as f4:
    clean_test_data = f4.readlines()


#进行训练
    
train_x = clean_train_data
#train_y = trans(train_label_list)
train_y = train_label_list
test_x = clean_test_data
#test_y = trans(test_label_list)
test_y = test_label_list

tfidf = TfidfVectorizer()
train_x = tfidf.fit_transform(train_x)

test_x = tfidf.transform(test_x)
t0 = time()
print('\t\t使用 TF-IDF 进行特征选择的朴素贝叶斯文本分类\t\t')
fit_and_predicted(train_x, train_y, test_x, test_y)
print('time uesed: %0.4fs' %(time() - t0))