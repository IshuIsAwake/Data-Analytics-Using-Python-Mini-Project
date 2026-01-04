import pandas as pd 
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 

class PopulationCentral:
    def __init__(self , nums):
        self.nums = nums 

    def mean(self):
        return self.nums.mean() 
    
    def std(self):
        return self.nums.std() 
    
    def variance(self): 
        return self.std() ** 2 

class SampleCentral: 
    def __init__(self,nums):
        self.nums = nums 

    def mean(self):
        return np.mean(self.nums)
    
    def variance(self): 
        meanVal = self.mean()
        s2 = 0
        for i in self.nums:
            s2 += (i - meanVal)**2 
        s2 /= len(self.nums) - 1 
        return s2 
    
    def std(self):
        return self.variance()**0.5 
    
class ZTest:
    def __init__(self, sigma, n, xbar=None, mu=None, z=None):
        self.sigma = sigma
        self.n = n

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}

        self.std_error = self.sigma / math.sqrt(self.n)

        if z is None:
            if xbar is None or mu is None:
                raise ValueError("To solve for 'z', you must provide 'xbar' and 'mu'.")
            self.xbar = xbar
            self.mu = mu
            self.z = (self.xbar - self.mu) / self.std_error
            
        elif xbar is None:
            if mu is None or z is None:
                raise ValueError("To solve for 'xbar', you must provide 'mu' and 'z'.")
            self.mu = mu
            self.z = z
            self.xbar = self.mu + (self.z * self.std_error)
            
        elif mu is None:
            if xbar is None or z is None:
                raise ValueError("To solve for 'mu', you must provide 'xbar' and 'z'.")
            self.xbar = xbar
            self.z = z
            self.mu = self.xbar - (self.z * self.std_error)
            
        else:
            self.xbar = xbar
            self.mu = mu
            self.z = z

            calculated_z = (self.xbar - self.mu) / self.std_error
            if not math.isclose(self.z, calculated_z):
                print(f"Warning: Provided values are inconsistent. {self.z} does not match {calculated_z}")

    def getZ(self):
        return self.z 

    def cdfZ(self):
        return stats.norm.cdf(self.getZ())
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfZ()
        elif tail == "right":
            return 1 - self.cdfZ()
        else:
            if self.getZ() >= 0:
                return 2*(1 - self.cdfZ())
            else: 
                return 2*(self.cdfZ())
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        
        if tail == "left":
            z = stats.norm.ppf(conf)
            answer = [self.xbar - z * self.std_error , math.inf]
            answer.sort()
            return answer
        elif tail == "right":
            z = stats.norm.ppf(conf)
            answer = [-math.inf , self.xbar + z*self.std_error]
            answer.sort()
            return answer
        else:
            z = stats.norm.ppf((1+conf)/2)
            answer = [self.xbar - z * self.std_error , self.xbar + z*self.std_error]
            answer.sort()
            return answer
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.mu < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_null=True, show_area=True, show_grid=True, x_span=5, xbins=40, ybins=20):
        se = self.std_error                        
        conf = confidence / 100.0
        ci = self.CI(confidence, tail)              
        center = self.xbar                           

        x_min = center - x_span * se
        x_max = center + x_span * se
        x = np.linspace(x_min, x_max, 1000)

        y_ci = stats.norm.pdf(x, loc=center, scale=se)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y_ci, lw=2, label="Sampling distribution")

        if show_null and hasattr(self, "mu") and self.mu is not None:
            y_null = stats.norm.pdf(x, loc=self.mu, scale=se)
            plt.plot(x, y_null, lw=1.5, linestyle="--", label="Null distribution")

        plt.axvline(self.xbar, linestyle="--", label=f"Sample Mean = {self.xbar:.2f}")
        if hasattr(self, "mu") and self.mu is not None:
            plt.axvline(self.mu, linestyle=":",color = "g" ,  label=f"Hypothesized Mean = {self.mu:.2f}")

        #shading
        if show_area:
            if tail == "both":
                a, b = ci
                mask = (x >= a) & (x <= b)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                _, upper = ci
                mask = (x <= upper)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
            elif tail == "right":
                lower, _ = ci
                mask = (x >= lower)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
        if tail == "left": 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
        elif tail == "right": 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}') 
        else: 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}')
        plt.title(f"Confidence Interval Visualization" , self.title)
        plt.xlabel("x" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
class ZTestDifference:

    def __init__(self, sigma1, n1, sigma2, n2, xbar_diff=None, mu_diff=None, z=None):
        self.sigma1 = sigma1
        self.n1 = n1
        self.sigma2 = sigma2
        self.n2 = n2

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}

        self.std_error = math.sqrt((self.sigma1**2 / self.n1) + (self.sigma2**2 / self.n2))

        if z is None:
            if xbar_diff is None or mu_diff is None:
                raise ValueError("To solve for 'z', you must provide 'xbar_diff' and 'mu_diff'.")
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.z = (self.xbar_diff - self.mu_diff) / self.std_error
            
        elif xbar_diff is None:
            if mu_diff is None or z is None:
                raise ValueError("To solve for 'xbar_diff', you must provide 'mu_diff' and 'z'.")
            self.mu_diff = mu_diff
            self.z = z
            self.xbar_diff = self.mu_diff + (self.z * self.std_error)
            
        elif mu_diff is None:
            if xbar_diff is None or z is None:
                raise ValueError("To solve for 'mu_diff', you must provide 'xbar_diff' and 'z'.")
            self.xbar_diff = xbar_diff
            self.z = z
            self.mu_diff = self.xbar_diff - (self.z * self.std_error)
            
        else:
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.z = z

            calculated_z = (self.xbar_diff - self.mu_diff) / self.std_error
            if not math.isclose(self.z, calculated_z):
                print(f"Warning: Provided values are inconsistent. {self.z} does not match {calculated_z}")

    def getZ(self):
        return self.z 

    def cdfZ(self):
        return stats.norm.cdf(self.getZ())
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfZ()
        elif tail == "right":
            return 1 - self.cdfZ()
        else:
            if self.getZ() >= 0:
                return 2*(1 - self.cdfZ())
            else: 
                return 2*(self.cdfZ())
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        
        if tail == "left":
            z = stats.norm.ppf(conf)
            answer = [-math.inf, self.xbar_diff + z * self.std_error]
            return answer
        elif tail == "right":
            z = stats.norm.ppf(conf)
            answer = [self.xbar_diff - z * self.std_error, math.inf]
            return answer
        else:
            z = stats.norm.ppf((1+conf)/2)
            answer = [self.xbar_diff - z * self.std_error , self.xbar_diff + z*self.std_error]
            answer.sort()
            return answer
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.mu_diff < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_null=True, show_area=True, show_grid=True, x_span=5, xbins=40, ybins=20):
        se = self.std_error                        
        conf = confidence / 100.0
        ci = self.CI(confidence, tail)              
        center = self.xbar_diff                           

        x_min = center - x_span * se
        x_max = center + x_span * se
        x = np.linspace(x_min, x_max, 1000)

        y_ci = stats.norm.pdf(x, loc=center, scale=se)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y_ci, lw=2, label="Sampling distribution of diff")

        if show_null and hasattr(self, "mu_diff") and self.mu_diff is not None:
            y_null = stats.norm.pdf(x, loc=self.mu_diff, scale=se)
            plt.plot(x, y_null, lw=1.5, linestyle="--", label="Null distribution")

        plt.axvline(self.xbar_diff, linestyle="--", label=f"Sample Diff = {self.xbar_diff:.2f}")
        if hasattr(self, "mu_diff") and self.mu_diff is not None:
            plt.axvline(self.mu_diff, linestyle=":",color = "g" ,  label=f"Hypothesized Diff = {self.mu_diff:.2f}")

        if show_area:
            if tail == "both":
                a, b = ci
                mask = (x >= a) & (x <= b)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                _, upper = ci
                mask = (x <= upper)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
            elif tail == "right":
                lower, _ = ci
                mask = (x >= lower)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
        
        if tail == "left": 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Upper Bound = {ci[1]:.2f}') 
        elif tail == "right": 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Lower Bound = {ci[0]:.2f}') 
        else: 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}')
            
        plt.title(f"Confidence Interval Visualization (Difference of Means)" , self.title)
        plt.xlabel("Difference in means" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

class TTest:
    def __init__(self, s, n, xbar=None, mu=None, t=None):
        self.s = s
        self.n = n
        self.df = n - 1

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}

        self.std_error = self.s / math.sqrt(self.n)

        if t is None:
            if xbar is None or mu is None:
                raise ValueError("To solve for 't', you must provide 'xbar' and 'mu'.")
            self.xbar = xbar
            self.mu = mu
            self.t = (self.xbar - self.mu) / self.std_error
            
        elif xbar is None:
            if mu is None or t is None:
                raise ValueError("To solve for 'xbar', you must provide 'mu' and 't'.")
            self.mu = mu
            self.t = t
            self.xbar = self.mu + (self.t * self.std_error)
            
        elif mu is None:
            if xbar is None or t is None:
                raise ValueError("To solve for 'mu', you must provide 'xbar' and 't'.")
            self.xbar = xbar
            self.t = t
            self.mu = self.xbar - (self.t * self.std_error)
            
        else:
            self.xbar = xbar
            self.mu = mu
            self.t = t

            calculated_t = (self.xbar - self.mu) / self.std_error
            if not math.isclose(self.t, calculated_t):
                print(f"Warning: Provided values are inconsistent. {self.t} does not match {calculated_t}")

    def getT(self):
        return self.t 

    def cdfT(self):
        return stats.t.cdf(self.getT(), df=self.df)
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfT()
        elif tail == "right":
            return 1 - self.cdfT()
        else:
            if self.getT() >= 0:
                return 2*(1 - self.cdfT())
            else: 
                return 2*(self.cdfT())
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        
        if tail == "left":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [-math.inf, self.xbar + t_crit * self.std_error]
            return answer
        elif tail == "right":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [self.xbar - t_crit * self.std_error, math.inf]
            return answer
        else:
            t_crit = stats.t.ppf((1+conf)/2, df=self.df)
            answer = [self.xbar - t_crit * self.std_error , self.xbar + t_crit*self.std_error]
            answer.sort()
            return answer
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.mu < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_null=True, show_area=True, show_grid=True, x_span=5, xbins=40, ybins=20):
        se = self.std_error                        
        conf = confidence / 100.0
        ci = self.CI(confidence, tail)              
        center = self.xbar                           

        x_min = center - x_span * se
        x_max = center + x_span * se
        x = np.linspace(x_min, x_max, 1000)

        y_ci = stats.t.pdf(x, loc=center, scale=se, df=self.df)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y_ci, lw=2, label=f"Sampling distribution (t, df={self.df})")

        if show_null and hasattr(self, "mu") and self.mu is not None:
            y_null = stats.t.pdf(x, loc=self.mu, scale=se, df=self.df)
            plt.plot(x, y_null, lw=1.5, linestyle="--", label="Null distribution")

        plt.axvline(self.xbar, linestyle="--", label=f"Sample Mean = {self.xbar:.2f}")
        if hasattr(self, "mu") and self.mu is not None:
            plt.axvline(self.mu, linestyle=":",color = "g" ,  label=f"Hypothesized Mean = {self.mu:.2f}")

        if show_area:
            if tail == "both":
                a, b = ci
                mask = (x >= a) & (x <= b)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                _, upper = ci
                mask = (x <= upper)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
            elif tail == "right":
                lower, _ = ci
                mask = (x >= lower)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")

        if tail == "left": 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Upper Bound = {ci[1]:.2f}') 
        elif tail == "right": 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Lower Bound = {ci[0]:.2f}') 
        else: 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}')
            
        plt.title(f"T-Test Confidence Interval Visualization" , self.title)
        plt.xticks(rotation = 90)
        plt.xlabel("x" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

class TTestPooled:
    def __init__(self, s1, n1, s2, n2, xbar_diff=None, mu_diff=None, t=None):
        self.s1 = s1
        self.n1 = n1
        self.s2 = s2
        self.n2 = n2
        self.df = n1 + n2 - 2

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}


        s_p_squared = ( ((n1 - 1) * s1**2) + ((n2 - 1) * s2**2) ) / self.df
        self.s_p = math.sqrt(s_p_squared)
        
        self.std_error = self.s_p * math.sqrt((1 / n1) + (1 / n2))

        if t is None:
            if xbar_diff is None or mu_diff is None:
                raise ValueError("To solve for 't', you must provide 'xbar_diff' and 'mu_diff'.")
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.t = (self.xbar_diff - self.mu_diff) / self.std_error
            
        elif xbar_diff is None:
            if mu_diff is None or t is None:
                raise ValueError("To solve for 'xbar_diff', you must provide 'mu_diff' and 't'.")
            self.mu_diff = mu_diff
            self.t = t
            self.xbar_diff = self.mu_diff + (self.t * self.std_error)
            
        elif mu_diff is None:
            if xbar_diff is None or t is None:
                raise ValueError("To solve for 'mu_diff', you must provide 'xbar_diff' and 't'.")
            self.xbar_diff = xbar_diff
            self.t = t
            self.mu_diff = self.xbar_diff - (self.t * self.std_error)
            
        else:
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.t = t

            calculated_t = (self.xbar_diff - self.mu_diff) / self.std_error
            if not math.isclose(self.t, calculated_t):
                print(f"Warning: Provided values are inconsistent. {self.t} does not match {calculated_t}")

    def getT(self):
        return self.t 

    def cdfT(self):
        return stats.t.cdf(self.getT(), df=self.df)
    
    def getpooled(self):
        return self.s_p
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfT()
        elif tail == "right":
            return 1 - self.cdfT()
        else:
            if self.getT() >= 0:
                return 2*(1 - self.cdfT())
            else: 
                return 2*(self.cdfT())
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        
        if tail == "left":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [-math.inf, self.xbar_diff + t_crit * self.std_error]
            return answer
        elif tail == "right":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [self.xbar_diff - t_crit * self.std_error, math.inf]
            return answer
        else:
            t_crit = stats.t.ppf((1+conf)/2, df=self.df)
            answer = [self.xbar_diff - t_crit * self.std_error , self.xbar_diff + t_crit * self.std_error]
            answer.sort()
            return answer
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.mu_diff < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_null=True, show_area=True, show_grid=True, x_span=5, xbins=40, ybins=20):
        se = self.std_error                        
        conf = confidence / 100.0
        ci = self.CI(confidence, tail)              
        center = self.xbar_diff                           

        x_min = center - x_span * se
        x_max = center + x_span * se
        x = np.linspace(x_min, x_max, 1000)

        y_ci = stats.t.pdf(x, loc=center, scale=se, df=self.df)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y_ci, lw=2, label=f"Sampling distribution of diff (t, df={self.df})")

        if show_null and hasattr(self, "mu_diff") and self.mu_diff is not None:
            y_null = stats.t.pdf(x, loc=self.mu_diff, scale=se, df=self.df)
            plt.plot(x, y_null, lw=1.5, linestyle="--", label="Null distribution")

        plt.axvline(self.xbar_diff, linestyle="--", label=f"Sample Diff = {self.xbar_diff:.2f}")
        if hasattr(self, "mu_diff") and self.mu_diff is not None:
            plt.axvline(self.mu_diff, linestyle=":",color = "g" ,  label=f"Hypothesized Diff = {self.mu_diff:.2f}")

        if show_area:
            if tail == "both":
                a, b = ci
                mask = (x >= a) & (x <= b)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                _, upper = ci
                mask = (x <= upper)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
            elif tail == "right":
                lower, _ = ci
                mask = (x >= lower)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
        
        if tail == "left": 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Upper Bound = {ci[1]:.2f}') 
        elif tail == "right": 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Lower Bound = {ci[0]:.2f}') 
        else: 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}')
            
        plt.title(f"Pooled T-Test CI Visualization (Difference of Means)" , self.title)
        plt.xlabel("Difference in means" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

class TTesteWelch:

    def __init__(self, s1, n1, s2, n2, xbar_diff=None, mu_diff=None, t=None):
        self.s1 = s1
        self.n1 = n1
        self.s2 = s2
        self.n2 = n2

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}
        
        self.var1_n = (self.s1**2) / self.n1
        self.var2_n = (self.s2**2) / self.n2

        self.std_error = math.sqrt(self.var1_n + self.var2_n)

        #Welch-Satterthwaite equation 
        self.df = (self.var1_n + self.var2_n)**2 / ( (self.var1_n**2 / (self.n1 - 1)) + (self.var2_n**2 / (self.n2 - 1)) )

        if t is None:
            if xbar_diff is None or mu_diff is None:
                raise ValueError("To solve for 't', you must provide 'xbar_diff' and 'mu_diff'.")
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.t = (self.xbar_diff - self.mu_diff) / self.std_error
            
        elif xbar_diff is None:
            if mu_diff is None or t is None:
                raise ValueError("To solve for 'xbar_diff', you must provide 'mu_diff' and 't'.")
            self.mu_diff = mu_diff
            self.t = t
            self.xbar_diff = self.mu_diff + (self.t * self.std_error)
            
        elif mu_diff is None:
            if xbar_diff is None or t is None:
                raise ValueError("To solve for 'mu_diff', you must provide 'xbar_diff' and 't'.")
            self.xbar_diff = xbar_diff
            self.t = t
            self.mu_diff = self.xbar_diff - (self.t * self.std_error)
            
        else:
            self.xbar_diff = xbar_diff
            self.mu_diff = mu_diff
            self.t = t

            calculated_t = (self.xbar_diff - self.mu_diff) / self.std_error
            if not math.isclose(self.t, calculated_t):
                print(f"Warning: Provided values are inconsistent. {self.t} does not match {calculated_t}")

    def getT(self):
        return self.t 

    def cdfT(self):
        return stats.t.cdf(self.getT(), df=self.df)
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfT()
        elif tail == "right":
            return 1 - self.cdfT()
        else:
            if self.getT() >= 0:
                return 2*(1 - self.cdfT())
            else: 
                return 2*(self.cdfT())
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        
        if tail == "left":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [-math.inf, self.xbar_diff + t_crit * self.std_error]
            return answer
        elif tail == "right":
            t_crit = stats.t.ppf(conf, df=self.df)
            answer = [self.xbar_diff - t_crit * self.std_error, math.inf]
            return answer
        else:
            t_crit = stats.t.ppf((1+conf)/2, df=self.df)
            answer = [self.xbar_diff - t_crit * self.std_error , self.xbar_diff + t_crit * self.std_error]
            answer.sort()
            return answer
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.mu_diff < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_null=True, show_area=True, show_grid=True, x_span=5, xbins=40, ybins=20):
        se = self.std_error                        
        conf = confidence / 100.0
        ci = self.CI(confidence, tail)              
        center = self.xbar_diff                           

        x_min = center - x_span * se
        x_max = center + x_span * se
        x = np.linspace(x_min, x_max, 1000)

        y_ci = stats.t.pdf(x, loc=center, scale=se, df=self.df)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y_ci, lw=2, label=f"Sampling distribution of diff (t, df={self.df:.2f})")

        if show_null and hasattr(self, "mu_diff") and self.mu_diff is not None:
            y_null = stats.t.pdf(x, loc=self.mu_diff, scale=se, df=self.df)
            plt.plot(x, y_null, lw=1.5, linestyle="--", label="Null distribution")

        plt.axvline(self.xbar_diff, linestyle="--", label=f"Sample Diff = {self.xbar_diff:.2f}")
        if hasattr(self, "mu_diff") and self.mu_diff is not None:
            plt.axvline(self.mu_diff, linestyle=":",color = "g" ,  label=f"Hypothesized Diff = {self.mu_diff:.2f}")

        if show_area:
            if tail == "both":
                a, b = ci
                mask = (x >= a) & (x <= b)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                _, upper = ci
                mask = (x <= upper)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
            elif tail == "right":
                lower, _ = ci
                mask = (x >= lower)
                plt.fill_between(x[mask], y_ci[mask], alpha=0.25, label=f"{confidence:.0f}% one-sided CI")
        
        if tail == "left": 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Upper Bound = {ci[1]:.2f}') 
        elif tail == "right": 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Lower Bound = {ci[0]:.2f}') 
        else: 
            plt.axvline(ci[0], color='red', linestyle='--', label=f'Left tail = {ci[0]:.2f}') 
            plt.axvline(ci[1], color='red', linestyle='--', label=f'Right tail = {ci[1]:.2f}')
            
        plt.title(f"Welch's T-Test CI Visualization (Difference of Means)" , self.title)
        plt.xlabel("Difference in means" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

class Chi:
    def __init__(self, n, s=None, sigma=None, chi2=None):
        self.n = n
        self.df = n - 1

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}

        if chi2 is None:
            if s is None or sigma is None:
                raise ValueError("To solve for 'chi2', you must provide 's' and 'sigma'.")
            self.s = s
            self.sigma = sigma
            self.s_var = s**2
            self.sigma_var = sigma**2
            self.chi2 = (self.df * self.s_var) / self.sigma_var
            
        elif s is None:
            if sigma is None or chi2 is None:
                raise ValueError("To solve for 's', you must provide 'sigma' and 'chi2'.")
            self.sigma = sigma
            self.chi2 = chi2
            self.sigma_var = sigma**2
            self.s_var = (self.chi2 * self.sigma_var) / self.df
            self.s = math.sqrt(self.s_var)
            
        elif sigma is None:
            if s is None or chi2 is None:
                raise ValueError("To solve for 'sigma', you must provide 's' and 'chi2'.")
            self.s = s
            self.chi2 = chi2
            self.s_var = s**2
            self.sigma_var = (self.df * self.s_var) / self.chi2
            self.sigma = math.sqrt(self.sigma_var)
            
        else:
            self.s = s
            self.sigma = sigma
            self.chi2 = chi2
            self.s_var = s**2
            self.sigma_var = sigma**2

            calculated_chi2 = (self.df * self.s_var) / self.sigma_var
            if not math.isclose(self.chi2, calculated_chi2):
                print(f"Warning: Provided values are inconsistent. {self.chi2} does not match {calculated_chi2}")

    def getChi2(self):
        return self.chi2 

    def cdfChi2(self):
        return stats.chi2.cdf(self.getChi2(), df=self.df)
    
    def getp(self , tail = "both"):
        if tail == "left":
            return self.cdfChi2()
        elif tail == "right":
            return 1 - self.cdfChi2()
        else:
            p_left = self.cdfChi2()
            p_right = 1 - p_left
            return 2 * min(p_left, p_right)
            
    def CI (self , confidence , tail = "both"): 
        if not (0 < confidence < 100):
            raise ValueError(f"Input value must be between 0 and 100, but got {confidence}")
        
        conf = confidence/100
        alpha = 1 - conf
        
        numerator = self.df * self.s_var
        
        if tail == "left":
            chi2_crit = stats.chi2.ppf(alpha, df=self.df) 
            upper = numerator / chi2_crit
            return [0, upper]
        elif tail == "right":
            chi2_crit = stats.chi2.ppf(1 - alpha, df=self.df) 
            lower = numerator / chi2_crit
            return [lower, math.inf]
        else:

            chi2_lower = stats.chi2.ppf(1 - alpha/2, df=self.df) 
            chi2_upper = stats.chi2.ppf(alpha/2, df=self.df) 
            
            lower_bound = numerator / chi2_lower
            upper_bound = numerator / chi2_upper
            return [lower_bound, upper_bound]
        
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return f"Failed to reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        else: 
            return f"Reject H0. There is a {100*self.getp()}% chance for sample to be equally or more extreme given H0 is true."
        
    def checkCI (self , confidence = 95 , tail = "both"):
        ci = self.CI(confidence , tail)
        if (ci[0] < self.sigma_var < ci[1]): 
            return f"Failed to reject H0"
        else:
            return f"Reject H0"
        
    def plotCI(self, confidence=95, tail="both", show_area=True, show_grid=True, x_span=3, xbins=40, ybins=20):

        conf = confidence / 100.0
        alpha = 1.0 - conf
        ci_stat = self.CI(confidence, tail) 
        
        if tail == 'both':
            chi2_crit_lower = stats.chi2.ppf(1 - alpha/2, df=self.df)
            chi2_crit_upper = stats.chi2.ppf(alpha/2, df=self.df)
        elif tail == 'left':
            chi2_crit_lower = -math.inf 
            chi2_crit_upper = stats.chi2.ppf(alpha, df=self.df)
        elif tail == 'right':
            chi2_crit_lower = stats.chi2.ppf(1 - alpha, df=self.df)
            chi2_crit_upper = math.inf
            

        x_min = 0
        x_max = stats.chi2.ppf(0.999, df=self.df)

        if tail != 'right': x_max = max(x_max, chi2_crit_upper * 1.2)
        if tail != 'left': x_max = max(x_max, chi2_crit_lower * 1.2)
            
        x = np.linspace(x_min, x_max, 1000)
        y = stats.chi2.pdf(x, df=self.df)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y, lw=2, label=f"Chi2")

        plt.axvline(self.chi2, linestyle="--", label=f"Calculated Chi2 = {self.chi2:.2f}")

        if show_area:
            if tail == "both":
                mask = (x >= chi2_crit_upper) & (x <= chi2_crit_lower)
                plt.fill_between(x[mask], y[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "left":
                mask = (x <= chi2_crit_upper)
                plt.fill_between(x[mask], y[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")
            elif tail == "right":
                mask = (x >= chi2_crit_lower)
                plt.fill_between(x[mask], y[mask], alpha=0.25, label=f"{confidence:.0f}% CI area")

        if tail == "left": 
            plt.axvline(chi2_crit_upper, color='red', linestyle='--', label=f'Crit = {chi2_crit_upper:.2f}') 
        elif tail == "right": 
            plt.axvline(chi2_crit_lower, color='red', linestyle='--', label=f'Crit = {chi2_crit_lower:.2f}') 
        else: 
            plt.axvline(chi2_crit_upper, color='red', linestyle='--', label=f'Crit upper {chi2_crit_upper:.2f}') 
            plt.axvline(chi2_crit_lower, color='red', linestyle='--', label=f'Crit lower {chi2_crit_lower:.2f}')
            
        plt.title(f"Chi-Squared CI Visualization" , self.title)
        plt.xlabel("Chi2 Statistic" , self.label)
        plt.ylabel("Density" , self.label)
        if show_grid:
            plt.grid(True, alpha=0.6, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

class GoF:
    def __init__(self, f_obs, f_exp, ddof=0):
        self.f_obs = np.array(f_obs)
        self.f_exp = np.array(f_exp)
        self.ddof = ddof

        if len(self.f_obs) != len(self.f_exp):
            raise ValueError("Observed and Expected frequency lists must be the same length.")
        if not math.isclose(np.sum(self.f_obs), np.sum(self.f_exp)):
             print(f"Warning: Sum of observed frequencies ({np.sum(self.f_obs)}) does not equal "
                   f"sum of expected frequencies ({np.sum(self.f_exp)}).")

        self.k = len(self.f_obs) 
        self.df = self.k - 1 - self.ddof
        
        if self.df < 1:
            raise ValueError(f"Degrees of freedom must be at least 1. Got df={self.df} (k={self.k}, ddof={self.ddof})")

        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}
        

        self.chi2 = np.sum( (self.f_obs - self.f_exp)**2 / self.f_exp )

    def getChi2(self):
        return self.chi2 
    
    def getdf(self):
        return self.df

    def cdfChi2(self):
        return stats.chi2.cdf(self.getChi2(), df=self.df)
    
    def getp(self):

        return 1 - self.cdfChi2()
            
    def checkp (self , alpha): 
        if self.getp() > alpha/100:
            return (f"Failed to reject H0 (p={self.getp():.4f}). Data fits the expected distribution.")
        else: 
            return (f"Reject H0 (p={self.getp():.4f}). Data does NOT fit the expected distribution.")
        

class ANOVA_OneWay:

    def __init__(self, groups: dict):
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups.")
        
        self.title = {'family': 'serif','color': 'black','size': 20,'weight': 'bold'}
        self.label = {'family': 'sans-serif','color': 'darkblue','size': 14,'style': 'italic'}
        
        self.groups = groups
        self.k = len(groups)                     
        self.N = sum(len(v) for v in groups.values())  

        self.all_values = [x for g in groups.values() for x in g]
        self.grand_mean = sum(self.all_values) / self.N

        self.SSB = self._compute_SSB()
        self.SSW = self._compute_SSW()

        self.df_between = self.k - 1
        self.df_within = self.N - self.k

        self.MSB = self.SSB / self.df_between
        self.MSW = self.SSW / self.df_within

        self.F = self.MSB / self.MSW

    def _compute_SSB(self):
        SSB = 0
        for g in self.groups.values():
            n = len(g)
            mean_g = sum(g) / n
            SSB += n * (mean_g - self.grand_mean)**2
        return SSB

    def _compute_SSW(self):
        SSW = 0
        for g in self.groups.values():
            mean_g = sum(g) / len(g)
            SSW += sum((x - mean_g)**2 for x in g)
        return SSW

    def getF(self):
        
        return self.F

    def getp(self, tail="right"):
        p = 1 - stats.f.cdf(self.F, self.df_between, self.df_within)
        return p

    def checkp(self, alpha=5):
        p = self.getp()
        if p > alpha/100:
            return f"Failed to reject H0. p = {p:.4f}. Groups likely have equal means."
        else:
            return f"Reject H0. p = {p:.4f}. At least one group mean differs."

    def plotF(self, x_span=5, xbins=40, ybins=20):
        x_max = max(10, self.F * x_span)
        x = np.linspace(0, x_max, 1000)
        y = stats.f.pdf(x, self.df_between, self.df_within)

        plt.figure(figsize=(10, 6))
        plt.locator_params(axis="x", nbins=xbins)
        plt.locator_params(axis="y", nbins=ybins)

        plt.plot(x, y, lw=2, label=f"F({self.df_between}, {self.df_within}) Distribution")
        plt.axvline(self.F, color='red', linestyle='--', label=f"Observed F = {self.F:.2f}")

        plt.fill_between(x, y, where=(x >= self.F), color='red', alpha=0.3,label="Right-tail (p-value)")

        plt.title("ANOVA F-Test Distribution", self.title)
        plt.xlabel("F value",self.label)
        plt.ylabel("Density", self.label)
        plt.legend()
        plt.grid(alpha=0.6, linestyle='--')
        plt.tight_layout()
        plt.show()

