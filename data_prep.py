# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

  #Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
import warnings
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, precision_recall_curve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')
sns.set(style = 'white')

train = pd.read_csv('/Users/babakmodami/Desktop/credit-card-lead-pred/train.csv')
test = pd.read_csv('/Users/babakmodami/Desktop/credit-card-lead-pred/test.csv')

train.columns, test.columns

train.head(10)

train.dtypes

# As most of the features are categorical, check if test data has all categorical values same as train data.
# If any column has new values then we should be careful using that feature in training directly..

count=0
for col in train.columns:
    if col not in ['ID','Age','Vintage','Avg_Account_Balance','Is_Lead']:
        for val in test[col].unique():
            if val not in train[col].unique():
                print(col,val)
                count+=1

test_region_list=test['Region_Code'].tolist()

train=train[train['Region_Code'].isin(test_region_list)]

train.isnull().sum(), test.isnull().sum()

# Lets check if data is uniformly distributed between train and test. Also,if there are any outliers or not.

train.describe()

def UVA_outlier(data, var_group, include_outlier = True):
    '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables\n
  include_outlier : {bool} whether to include outliers or not, default = True\n
  '''
    size=len(var_group)
    plt.figure(figsize = (7*size,4), dpi = 100)

#looping for each variable
    for j,i in enumerate(var_group):
        # calculating descriptives of variable
        quant25 = data[i].quantile(0.25)
        quant75 = data[i].quantile(0.75)
        IQR = quant75 - quant25
        med = data[i].median()
        whis_low = med-(1.5*IQR)
        whis_high = med+(1.5*IQR)
  

# Calculating Number of Outliers
        outlier_high = len(data[i][data[i]>whis_high])
        outlier_low = len(data[i][data[i]<whis_low])

        if include_outlier == True:
            print(include_outlier)
        #Plotting the variable with every information
            plt.subplot(1,size,j+1)
            sns.boxplot(data[i], orient="v")
            plt.ylabel('{}'.format(i))
            plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlierhigh)
                                                                                                   ))
            # replacing outliers with max/min whisker
            train = data[var_group][:]
            train[i][train[i]>whis_high] = whis_high+1
            train[i][train[i]<whis_low] = whis_low-1
      
      # plotting without outliers
            plt.subplot(1, size,j+1)
            sns.boxplot(train[i], orient="v")
            plt.ylabel('{}'.format(i))
            plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   ))
num_cols = ['Age', 'Vintage','Avg_Account_Balance']

UVA_outlier(train, num_cols)

# There seems to be large no of outliers in variable "Avg_Account_Balance". Lets also check for outliers in test["Avr_Account_Balance"]
test.boxplot(column = ['Avg_Account_Balance'])

#Bivariate Analysis
#Continuius Continuous
numericals = train[['Age', 'Vintage','Avg_Account_Balance']]
correlation = numericals.corr()
correlation

plt.figure(figsize=(36,6), dpi=140)
for j,i in enumerate(['pearson','kendall','spearman']):
    plt.subplot(1,3,j+1)
    correlation = numericals.dropna().corr(method=i)
    sns.heatmap(correlation, linewidth = 2)
    plt.title(i, fontsize=18)
                                        

