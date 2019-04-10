#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import GistUtils
import numpy as np
import pandas as pd


labels = {0:'ACB', 1:'AnFpark', 2:'FDFpark'}


def generate_trainset():
    dataset_array = list()
    label_array = list()
    for i in range(80):
        img_name = 'SceneCategory/TrainData/ACB/ACB_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(0)
    for i in range(80):
        img_name = 'SceneCategory/TrainData/AnFpark/AnFpark_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(1)
    for i in range(80):
        img_name = 'SceneCategory/TrainData/FDFpark/FDFpark_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(2)

    dataset_df = pd.DataFrame(dataset_array)
    dataset_df['label'] = label_array
    dataset_df.to_csv('SceneCategory/trainset.csv', index=False)


def generate_testset():
    dataset_array = list()
    label_array = list()
    for i in range(80, 100):
        img_name = 'SceneCategory/TestData/ACB/ACB_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(0)
    for i in range(80, 100):
        img_name = 'SceneCategory/TestData/AnFpark/AnFpark_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(1)
    for i in range(80, 100):
        img_name = 'SceneCategory/TestData/FDFpark/FDFpark_%d.jpeg' % i
        img_gist = get_gist(img_name)
        dataset_array.append(img_gist)
        label_array.append(2)

    dataset_df = pd.DataFrame(dataset_array)
    dataset_df['label'] = label_array
    dataset_df.to_csv('SceneCategory/testset.csv', index=False)


def get_gist(filename):
    img = Image.open(filename)
    return GistUtils.get_gist(img, blocks=4, direction=8, scale=[5, 8, 11, 14])


def get_trainset():
    pass





