import pandas as pd
import numpy as np

import streamlit as st

import pathlib

# data and plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.precision", 3)
sns.set(style='ticks')

# scikit learn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, roc_curve, auc, mean_squared_log_error
from sklearn.metrics import roc_curve, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

# ----------------
# 1. load data
# ----------------

data_dir = pathlib.Path('./data')

# targets: (weight, regularization)
targets_info = {
  'age': (0.3, 100),
  'domain1_var1': (0.175, 10),
  'domain1_var2': (0.175, 10),
  'domain2_var1': (0.175, 10),
  'domain2_var2': (0.175, 10)
}

@st.cache
def load_data():

  fnc_df = pd.read_csv(data_dir / 'fnc.csv')
  loading_df = pd.read_csv(data_dir / 'loading.csv')
  sample_submission_df = pd.read_csv(data_dir / 'sample_submission.csv')

  fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
  # features and targets
  features = loading_features #+ fnc_features

  train_scores_df = pd.read_csv(data_dir / 'train_scores.csv')

  train_scores_df['is_train'] = True

  # combine all features and targets
  combined_df = train_scores_df.merge(loading_df, on='Id').merge(fnc_df, on='Id', how='left')

  combined_df.fillna(value={'is_train':False}, inplace=True)

  return combined_df, loading_features, fnc_features

#---------------------------------


combined_df, loading_features, fnc_features = load_data()
targets = list(targets_info.keys())

st.title('TReNDS Neuroimaging')

col = st.sidebar.selectbox('Select a column to summerize (X)', combined_df.columns)

if col:
  st.header('Column Summary (X)')
  st.write(combined_df[[col]].describe())

  st.subheader('Distribution')
  sns.distplot(combined_df[[col]])
  st.pyplot()

  col2 = st.sidebar.selectbox('Select a second column (Y)', combined_df.columns)

  if col2:
    st.header('X/Y Correlation')
    sns.jointplot(col, col2, data=combined_df, kind="reg", joint_kws = {'scatter_kws': {"alpha": 0.01}})
    st.pyplot()
#

st.header('Loading/Targets Clusters')
sns.clustermap(combined_df[loading_features + targets].corr())
st.pyplot()