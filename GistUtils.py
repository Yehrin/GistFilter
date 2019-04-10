#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from PIL import Image


def get_gist(img, blocks, direction, scale):
    """
    获得图片的 GIST 向量
    :param img: 图片数据
    :param blocks: 划分成 blocks*blocks个patch
    :param direction: garbo 滤波器的方向
    :param scale: 滤波器的尺度
    :return: gist特征
    """
    # 构建滤波器
    filters = build_filters(direction, scale)
    # 分割图片
    img_arr = crop_image(img, blocks)
    # gist向量
    img_array = gist_process(img_arr, filters)
    return img_array


def crop_image(img, blocks):
    """
    分割图片为 blocks x blocks 个小图片
    :param img: 为 PIL.Image.Image 类型
    :param blocks:
    :return: blocks x blocks 个 PIL.Image.Image 类型的 list
    """
    img_array = []
    w, h = img.size
    patch_with = w / blocks
    patch_height = h / blocks
    # 高
    for i in range(blocks):
        # 宽
        for j in range(blocks):
            crop_img = img.crop((j * patch_with, i * patch_height, patch_with * (j + 1), patch_height * (i + 1)))
            img_array.append(crop_img)
    return img_array


def build_filters(direction, scale):
    """
    构建 Gabor 滤波器
    :param direction: gabor 方向
    :param scale: gabor 尺度
    :return: 滤波器 list
    """
    filters = []
    # 波长
    lamda = np.pi / 2.0
    for theta in np.arange(0, np.pi, np.pi / direction):
        for k in range(len(scale)):
            kern = cv2.getGaborKernel((scale[k], scale[k]), 1.0, theta, lamda,
                                      0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def gist_process(img_array, filters):
    """
    对图片进行滤波，得到 GIST 向量
    :param img_array: PIL.Image.Image 类型的 list
    :param filters: 滤波器组
    :return:
    """
    result = []
    # 图片不同通道要使用不同的参数
    ddepth_dict = {1: cv2.CV_8UC1, 2: cv2.CV_8UC2, 3: cv2.CV_8UC3}
    for img in img_array:
        img_nparray = np.asarray(img)
        for filter in filters:
            accum = np.zeros_like(img_nparray)
            for kern in filter:
                filtered_img = cv2.filter2D(img_nparray, ddepth=ddepth_dict[img_nparray.shape[2]],
                                            kernel=kern)
                np.maximum(accum, filtered_img, accum)
            average = np.mean(accum)
            result.append(average)
    # 结果保留4位小数
    round_result = np.round(result, 4)
    return round_result


if __name__ == '__main__':
    img1 = Image.open('SceneCategory/TrainData/ACB/ACB_0.jpeg')
    print(get_gist(img1, blocks=4, direction=8, scale=[5, 8, 11, 14]))
