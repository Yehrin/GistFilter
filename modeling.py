#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
# import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
import time


def get_model(algorithm):
    """
    获得指定的模型
    :param algorithm: 可以是：'svm', 'knn', 'xgboost', 'decision_tree'
    :return:
    """
    if algorithm == 'svm':
        model = svm.LinearSVC(C=0.5, class_weight='balanced', max_iter=1e6, verbose=1)
    elif algorithm == 'knn':
        model = neighbors.KNeighborsClassifier()
    # elif algorithm == 'xgboost':
    #     model = xgboost.XGBClassifier()
    elif algorithm == 'decision_tree':
        model = tree.DecisionTreeClassifier(max_leaf_nodes=3)
    else:
        print('not algorithm')
        return None
    return model


def model_evaluate(model, testset_feature, testset_label):
    start_time = time.time()
    testset_predict = model.predict(testset_feature)
    stop_time = time.time()
    print('predict 耗时：%f s' % (stop_time - start_time))
    print('准确度 Accuracy：', accuracy_score(testset_label, testset_predict))
    print(classification_report(testset_label, testset_predict))
    # 绘制 ROC 曲线
    testset_score = model.decision_function(testset_feature)
    testset_auc = roc_auc_score(testset_label, testset_score)
    print('AUC: ', testset_auc)

    fpr, tpr, thresholds = roc_curve(testset_label, testset_score)
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('Model/roc_wave.png')
    return testset_predict, testset_score


if __name__ == '__main__':
    pass
