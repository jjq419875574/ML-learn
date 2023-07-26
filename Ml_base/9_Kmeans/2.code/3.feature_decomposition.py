# coding:utf-8

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


def var_thr():
    """
    特征选择:低方差特征过滤
    :return:
    """
    data = pd.read_csv("./data/factor_returns.csv")
    # print(data)
    print(data.shape)

    # 实例化一个对象
    transfer = VarianceThreshold(threshold=10)
    # 转换
    transfer_data = transfer.fit_transform(data.iloc[:, 1:10])
    print(transfer_data)
    print(data.iloc[:, 1:10].shape)
    print(transfer_data.shape)


def pea_demo():
    """
    皮尔逊相关系数
    :return:
    """
    # 准备数据
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

    # 判断
    ret = pearsonr(x1, x2)
    print("皮尔逊相关系数的结果是:\n", ret)


def spea_demo():
    """
    斯皮尔曼相关系数
    :return:
    """
    # 准备数据
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

    # 判断
    ret = spearmanr(x1, x2)
    print("斯皮尔曼相关系数的结果是:\n", ret)


def pca_demo():
    """
    pca降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # pca小数保留百分比
    transfer = PCA(n_components=0.9)
    trans_data = transfer.fit_transform(data)
    print("保留0.9的数据最后维度为:\n", trans_data)

    # pca小数保留百分比
    transfer = PCA(n_components=3)
    trans_data = transfer.fit_transform(data)
    print("保留三列数据:\n", trans_data)


if __name__ == '__main__':
    # var_thr()
    # pea_demo()
    # spea_demo()
    pca_demo()
