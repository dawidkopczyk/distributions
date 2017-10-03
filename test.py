import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stats.vg import vg
from stats.hyperb import hyperb

data = pd.read_csv('data.csv')
ticker = data.columns[1:3]
x = np.array(data[ticker])
chng = np.log(x[1:,:]/x[0:-1,:])

fig, axes = plt.subplots(nrows=2,ncols=2)
sns.distplot(chng[:,0],kde=False,fit=hyperb, ax=axes[0,0], hist_kws={'color':'g','alpha':0.2}, fit_kws={'color':'g'})
sns.distplot(chng[:,0],kde=False,fit=vg, ax=axes[0,1], hist_kws={'color':'g','alpha':0.2}, fit_kws={'color':'g'})
sns.distplot(chng[:,1],kde=False,fit=hyperb, ax=axes[1,0], hist_kws={'color':'b','alpha':0.2}, fit_kws={'color':'b'})
sns.distplot(chng[:,1],kde=False,fit=vg, ax=axes[1,1], hist_kws={'color':'b','alpha':0.2}, fit_kws={'color':'b'})
axes[0,0].set_title(ticker[0]+': Hyperbolic Fit')
axes[0,1].set_title(ticker[0]+': VarianceGamma Fit')
axes[1,0].set_title(ticker[1]+': Hyperbolic Fit')
axes[1,1].set_title(ticker[1]+': VarianceGamma Fit')
plt.tight_layout()

