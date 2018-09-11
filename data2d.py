#!/usr/bin/python
import numpy as np


# 2次元データのクラス
class Data2D:
    # 乱数で初期化
    def __init__(self, mu, cov, n):
        self.x, self.y = np.random.multivariate_normal(mu, cov, n).T

    # 行列としてデータを取得
    def get_data(self):
        return np.vstack([self.x, self.y]).T
