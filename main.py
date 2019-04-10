#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import GistUtils
import data_set
import modeling





if __name__ == '__main__':
    print('Begin...')
    # 图片转csv
    # data_set.generate_trainset()
    # data_set.generate_testset()

    # 读取数据集
    trainset_df = pd.read_csv('SceneCategory/trainset.csv')
    testset_df = pd.read_csv('SceneCategory/testset.csv')
    # 数据清洗
    # 特征组合
    # 提取特征和标签
    trainset_features = trainset_df.drop(labels=['label'], axis=1)
    trainset_label = trainset_df['label']

    testset_features = testset_df.drop(labels=['label'], axis=1)
    testset_label = testset_df['label']

    # 选择模型
    sklearn_model = modeling.get_model('svm')
    # 交叉验证
    score = cross_val_score(sklearn_model, X=trainset_features, y=trainset_label, cv=10)  # 10折
    print('交叉编译的准确度：', score)
    # 训练模型
    sklearn_model.fit(trainset_features, trainset_label)
    testset_predicted_label = sklearn_model.predict(testset_features)




