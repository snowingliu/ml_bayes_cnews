# 机器学习-贝叶斯

## 程序说明

### clean.py

```p
1.读取文件
2.用jieba分词进行文本分割
3.将清洗好的文件存入新的文档
```

### by.py

```python
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
```

## 预测结果

使用 TF-IDF 进行特征选择的朴素贝叶斯文本分类              
​                                   precision    recall          f1-score        support

```
     体育       1.00      1.00      1.00      1000
     娱乐       0.92      0.99      0.96      1000
     家居       0.97      0.30      0.46      1000
     房产       0.58      0.93      0.72      1000
     教育       0.90      0.94      0.92      1000
     时尚       0.98      0.96      0.97      1000
     时政       0.96      0.89      0.92      1000
     游戏       0.97      0.97      0.97      1000
     科技       0.95      0.99      0.97      1000
     财经       0.96      0.99      0.98      1000
```

avg / total       0.92      0.90      0.89     10000

accuracy_score: 0.89680s
time uesed: 0.4278s
