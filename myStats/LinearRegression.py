import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

class LinearReg:
    def __init__(self, xvals, yvals):
        x = np.asarray(xvals, dtype=float).ravel()
        y = np.asarray(yvals, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape (both 1-D).")
        n = x.size
        if n < 2:
            raise ValueError("Need at least 2 points for linear regression.")

        self.xvals = x
        self.yvals = y
        self.n = n
        self.xbar = x.mean()
        self.ybar = y.mean()

        xc = x - self.xbar
        yc = y - self.ybar

        Sxx = np.sum(xc * xc)
        Sxy = np.sum(xc * yc)
        if Sxx == 0:
            raise ValueError("All x values are identical; slope is undefined.")

        self.beta1 = Sxy / Sxx
        self.beta0 = self.ybar - self.beta1 * self.xbar

        self.ycap = self.beta0 + self.beta1 * x
        self.residuals = y - self.ycap

        self.SSres = float(np.sum(self.residuals ** 2))              
        self.SStot = float(np.sum((y - self.ybar) ** 2))             
        self.SSreg = float(np.sum((self.ycap - self.ybar) ** 2))     

        if self.SStot == 0:
            self.R2 = 1.0 if np.allclose(self.ycap, y) else 0.0
        else:
            self.R2 = 1.0 - (self.SSres / self.SStot)

        self.df_resid = max(n - 2, 0)
        if self.df_resid > 0:
            self.MSE = self.SSres / self.df_resid
            self.SE_beta1 = math.sqrt(self.MSE / Sxx)
            self.SE_beta0 = math.sqrt(self.MSE * (1.0/n + (self.xbar**2) / Sxx))
            self.t_beta1 = self.beta1 / self.SE_beta1
            self.t_beta0 = self.beta0 / self.SE_beta0
            self.p_beta1 = 2 * (1 - stats.t.cdf(abs(self.t_beta1), df=self.df_resid))
            self.p_beta0 = 2 * (1 - stats.t.cdf(abs(self.t_beta0), df=self.df_resid))
        else:
            self.MSE = self.SE_beta1 = self.SE_beta0 = None
            self.t_beta1 = self.t_beta0 = None
            self.p_beta1 = self.p_beta0 = None

        self.df = pd.DataFrame({
            "Xi": x,
            "Yi": y,
            "Ycap": self.ycap,
            "Residual": self.residuals
        })

        self.rmse_ = float(np.sqrt(np.mean(self.residuals**2)))
        self.mae_  = float(np.mean(np.abs(self.residuals)))

        self.title = {'family': 'serif', 'color': 'black', 'size': 20, 'weight': 'bold'}
        self.label = {'family': 'sans-serif', 'color': 'darkblue', 'size': 14, 'style': 'italic'}

    def lineOfRegression(self, x):
        return self.beta0 + self.beta1 * x

    def predict(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        return self.beta0 + self.beta1 * x_new

    def Gof(self):
        return self.R2

    def RMSE(self):
        return self.rmse_

    def MAE(self):
        return self.mae_

    def summary(self):
        lines = [
            f"n = {self.n}, df_resid = {self.df_resid}",
            f"xbar = {self.xbar:.2f} , ybar = {self.ybar:.2f}",
            f"beta0 (intercept) = {self.beta0:.6g}",
            f"beta1 (slope)     = {self.beta1:.6g}",
            f"R^2               = {self.R2:.6g}",
            f"SStot={self.SStot:.6g}, SSreg={self.SSreg:.6g}, SSres={self.SSres:.6g}",
            f"RMSE              = {self.rmse_:.6g}",
            f"MAE               = {self.mae_:.6g}",
        ]
        if self.df_resid > 0:
            lines += [
                f"MSE (RSS/df)      = {self.MSE:.6g}",
                f"SE(beta0)         = {self.SE_beta0:.6g}, t={self.t_beta0:.6g}, p={self.p_beta0:.3g}",
                f"SE(beta1)         = {self.SE_beta1:.6g}, t={self.t_beta1:.6g}, p={self.p_beta1:.3g}",
            ]
        for i in lines:
            print(i + "\n")

    def LoRPlot(self, show_grid=True, xbins=40, ybins=20):
        order = np.argsort(self.xvals)
        xs = self.xvals[order]
        yhat = self.ycap[order]

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)
        plt.plot(self.xvals, self.yvals, "og", label="Actual Y")
        plt.plot(xs, yhat, "-", label="Predicted Y")
        plt.axhline(self.ybar, linestyle="--" , color = "red", label="Mean Y")
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.title("Linear Regression", self.title)
        plt.xlabel("x", self.label)
        plt.ylabel("y", self.label)
        plt.xticks(rotation = 90)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def residual_plots(self, show_grid=True, xbins=40, ybins=20):
        #Residuals vs Fitted
        plt.figure(figsize=(8, 5))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)
        plt.xticks(rotation = 90)
        plt.axhline(0, linestyle="--")
        plt.plot(self.ycap, self.residuals, "o")
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.title("Residuals vs Fitted", self.title)
        plt.xlabel("Fitted values $\hat{y}$", self.label)
        plt.ylabel("Residuals", self.label)
        plt.tight_layout()
        plt.show()

        #Residuals vs X
        plt.figure(figsize=(8, 5))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)
        plt.xticks(rotation = 90)
        plt.axhline(0, linestyle="--")
        plt.plot(self.xvals, self.residuals, "o")
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.title("Residuals vs X", self.title)
        plt.xlabel("X", self.label)
        plt.ylabel("Residuals", self.label)
        plt.tight_layout()
        plt.show()

        #Quantile Quantile plot
        plt.figure(figsize=(8, 5))
        stats.probplot(self.residuals, dist="norm", plot=plt)
        plt.xticks(rotation = 90)
        plt.title("QQ-plot of Residuals", self.title)
        plt.tight_layout()
        plt.show()
