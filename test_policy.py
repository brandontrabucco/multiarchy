"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.distributions.tanh_gaussian import TanhGaussian
from multiarchy.distributions.gaussian import Gaussian
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    for std in [0.01, 0.1, 0.5, 1.0]:
        num_divisions = 1000
        policy = Gaussian(lambda batch: np.ones([*batch.shape, 2]), std=std)
        x = np.linspace(-1.0, 1.0, num=num_divisions)
        x = np.stack(np.meshgrid(x, x), -1)
        img = np.exp(policy.log_prob(x, np.zeros([1])).numpy())
        plt.imshow(img)
        plt.show()

    for std in [0.01, 0.1, 0.5, 1.0]:
        num_divisions = 1000
        policy = TanhGaussian(lambda batch: np.ones([*batch.shape, 2]), std=std)
        x = np.linspace(-1.0, 1.0, num=num_divisions)
        x = np.stack(np.meshgrid(x, x), -1)
        img = np.exp(policy.log_prob(x.astype(np.float32), np.zeros([1])).numpy())
        plt.imshow(img)
        plt.show()
