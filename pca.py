import numpy as np


# データを正規化
def normalize(X):
    mn = np.mean(X, axis=0)
    z = X - mn
    return z


# pcaのクラス
class PCA:
    def fit(self, X):
        z = normalize(X)
        cv = np.cov(z[:, 0], z[:, 1], bias=1)
        self.W, self.v = np.linalg.eig(cv)

    # 方向ベクトルを取得
    def get_vec(self):
        return self.v[:, np.argmax(self.W)]

    # 寄与率を取得
    def get_contribution_rate(self):
        return np.max(self.W)/np.sum(self.W)
