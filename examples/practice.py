from pathlib import Path

import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

from data import get_moon_data
from examples import RBF_kernel_fn, linear_kernel_fn, train_kernel_svm_classifier_with_gram
from utils import plot_decision_boundary, plot_data

Path("./results").mkdir(parents=True, exist_ok=True)

data_name = "moon_data"
data_x, data_y = get_moon_data()


def demo_plot_data():
    """
    plot distribution of datasets
    """
    plot_data(data_x, data_y, title=data_name,
              path="./results/{}.jpg".format(data_name))


def demo_linear_svm():
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=linear_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=linear_kernel_fn, boundary=True,
                           title="linear kernel prediction",
                           path="./results/linear_{}.jpg".format(data_name))


def demo_rbf_svm():
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=RBF_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=RBF_kernel_fn, boundary=True,
                           title="rbf kernel prediction",
                           path="./results/rbf_{}.jpg".format(data_name))


def RFF_kernel_fn(x1, x2):
    """
    Approximate RBF kernel with random fourier features.
    Reference:
        Random Features for Large-Scale Kernel Machines
        https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Input: 
        x1.shape=(x1_n, d)
        x2.shape=(x2_n, d)
    Return:
        gram.shape=(x1_n, x2_n)

    TODO: Complete this function.
    """
    pass


def test_RFF_kernel_fn():
    """
    TODO:
        1. investigate how the dimension of random fourier features affect the precision of approximation.
        2. investigate how x_dim affect the speed of rbf kernel and rff kernel.

    Reference:
        On the Error of Random Fourier Features, UAI 2015
        https://arxiv.org/abs/1506.02785
    """
    x_dim = 100
    x1 = np.random.randn(x_dim, 2)
    x2 = np.random.randn(x_dim, 2)
    gram_rbf = RBF_kernel_fn(x1, x2)
    gram_rff = RFF_kernel_fn(x1, x2)
    # diff = np.max(np.abs(gram_rbf - gram_rff))
    diff = np.mean((gram_rbf - gram_rff)**2)
    print("MSE of gram matrix: {:.10f}".format(diff))
    # D=100000, MSE ≈ 1e-5


def test_RFF_kernel_svm():
    """Test how your RFF perform.
    """
    pass

if __name__ == "__main__":
    demo_plot_data()
    demo_linear_svm()
    demo_rbf_svm()