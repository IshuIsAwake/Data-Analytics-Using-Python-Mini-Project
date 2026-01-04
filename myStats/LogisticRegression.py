import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

class LogisticReg:
    def __init__(self, xvals, yvals):
        x = np.asarray(xvals, dtype=float).ravel()
        y = np.asarray(yvals, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be binary")
        self.x = x
        self.y = y
        self.beta0 = 0.0
        self.beta1 = 0.0
        for _ in range(5000): #5000 steps
            z = self.beta0 + self.beta1 * x
            p = 1 / (1 + np.exp(-z))
            self.beta0 -= 0.05 * np.mean(p - y) #learning rate is 0.05 for training beta0 and beta1
            self.beta1 -= 0.05 * np.mean((p - y) * x)
        self.proba = 1 / (1 + np.exp(-(self.beta0 + self.beta1 * x)))

    def predict_prob(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        z = self.beta0 + self.beta1 * x_new
        return 1 / (1 + np.exp(-z))

    def predict(self, x_new):
        return (self.predict_prob(x_new) >= 0.5).astype(int)

    def summary(self):
        print(f"beta0 (intercept) = {self.beta0:.6g}")
        print(f"beta1 (slope)     = {self.beta1:.6g}")
