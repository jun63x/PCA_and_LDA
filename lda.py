import numpy as np
import matplotlib.pyplot as plt


# ldaのクラス
class LDA:
    def fit(self, X1, X2):
        self.mn1 = np.mean(X1, axis=0)
        self.mn2 = np.mean(X2, axis=0)
        SW_size = self.mn1.shape[0]
        self.SW = np.zeros((SW_size, SW_size))
        for row1, row2 in zip(X1, X2):
            v1 = (row1 - self.mn1).reshape(SW_size, 1)
            v2 = (row2 - self.mn2).reshape(SW_size, 1)
            self.SW += np.dot(v1, v1.T) + np.dot(v2, v2.T)

    # 方向ベクトルを取得
    def get_vec(self):
        return np.dot(np.linalg.inv(self.SW), self.mn1 - self.mn2)
