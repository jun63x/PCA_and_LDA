import numpy as np
import matplotlib.pyplot as plt
from data2d import Data2D
from pca import PCA
from lda import LDA


# データを射影
def project(mat, vector):
    return [np.dot(row, vector) for row in mat]


# ヒストグラムを表示
def show_hist(X1, X2, fig_name=None):
    plt.hist(X1, color='red', alpha=0.5)
    plt.hist(X2, color='blue', alpha=0.5)
    if fig_name is not None:
        plt.savefig(fig_name, format='png', dpi=200)
    plt.show()
    plt.close()


if __name__ == '__main__':
    mu1, mu2 = [3, 1], [1, 3]  # 平均
    cov = [[1, 2], [2, 5]]  # 共分散
    n = 10000  # データ数
    data1 = Data2D(mu1, cov, n)
    data2 = Data2D(mu2, cov, n)
    X1 = data1.get_data()
    X2 = data2.get_data()
    X = np.vstack([X1, X2])

    # PCA
    pca = PCA()
    pca.fit(X)
    pca_vec = pca.get_vec()
    show_hist(project(X1, pca_vec), project(X2, pca_vec))

    # LDA
    lda = LDA()
    lda.fit(X1, X2)
    lda_vec = lda.get_vec()
    show_hist(project(X1, lda_vec), project(X2, lda_vec))

    # グラフ描画
    # 背景を白にする
    plt.figure(facecolor="w")

    axis_x = np.linspace(-10, 10)
    pca_y = (pca_vec[1] / pca_vec[0]) * axis_x
    lda_y = (lda_vec[0] / lda_vec[1]) * axis_x
    plt.plot(axis_x, pca_y, "c-", label="PCA")
    plt.plot(axis_x, lda_y, "m-", label="LDA")

    # 散布図をプロットする
    plt.scatter(data1.x, data1.y, color='r', marker='x')
    plt.scatter(data2.x, data2.y, color='b', marker='x')

    # ラベル
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20)

    # 軸
    plt.axis([-5.0, 10.0, -10.0, 15.0], size=20)
    plt.grid(True)
    plt.legend()

    # 保存
    # plt.savefig("pca_and_lda.png", format = 'png', dpi=200)
    plt.show()
    plt.close()
