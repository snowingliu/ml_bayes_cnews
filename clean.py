#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:36:10 2018

@author: liuxueying
"""


import jieba
import re



#读取词表
with open('cnews/cnews.train.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    #读取每行
    train_label_list = [k.split()[0] for k in line_list]
    #将标签依次取出
    train_content_list = [k.split(maxsplit=1)[1] for k in line_list]
    #将内容依次取出,此处注意split()选择最大分割次数为1,否则句子被打断.
#同理读取test数据
with open('cnews/cnews.test.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    test_label_list = [k.split()[0] for k in line_list]
    test_content_list = [k.split(maxsplit=1)[1] for k in line_list]

#使用jieba分词进行文本分割
def cutWord(content_list):
    cut_list = []
    for i in range(len(content_list)):
        a = ' '.join(jieba.cut(re.sub('[  \n\r\t]+', '', content_list[i])))
        cut_list.append(a)
    return cut_list



clean_train_content = cutWord(train_content_list)
clean_test_content = cutWord(test_content_list)

#将清洗好的数据放到新的文件中
with open('cnews/data/clean_train_data.txt', 'w') as f1:
    for essay in clean_train_content:
        f1.write(essay + '\n')

with open('cnews/data/clean_test_data.txt', 'w') as f2:
    for essay in clean_test_content:
        f2.write(essay + '\n')
        





