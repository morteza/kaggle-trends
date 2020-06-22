import pandas as pd
import numpy as np

import streamlit as st

import pathlib

# data and plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(context='notebook')

pd.set_option("display.precision", 3)

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
  combined_df = fnc_df.merge(loading_df, on='Id').merge(train_scores_df, on='Id', how='left')

  combined_df.fillna(value={'is_train':False}, inplace=True)

  return combined_df, loading_features, fnc_features

#---------------------------------


combined_df, loading_features, fnc_features = load_data()
target_features = list(targets_info.keys())


def summarize_feature(feats):
  st.header(f'Feature Summary')
  st.write(combined_df[feats].describe())

  st.subheader(f'Distribution')
  for f in feats:
    sns.distplot(combined_df[[f]])
  plt.legend(labels=feats)
  plt.tight_layout()
  st.pyplot()


st.title('TReNDS Neuroimaging EDA (May 15)')

##########
# features
st.sidebar.header(f'Features ({combined_df.Id.nunique()} subjects)')
show_sbm = st.sidebar.checkbox(f'SBM ({len(loading_features)} components)')
show_fnc = st.sidebar.checkbox(f'FNC ({len(fnc_features)} features)')
show_sm = st.sidebar.checkbox(f'Spatial Maps (53 components)')
show_targets = st.sidebar.checkbox(f'Targets ({len(target_features)} features)')

##########



if show_sbm:
  st.header('SBM')

  st.subheader('Clusters')
  st.markdown('Note: includes both test and training datasets')
  feats = loading_features + (target_features if show_targets else [])
  g = sns.clustermap(combined_df[feats].corr(), cmap='Blues')
  st.pyplot()

  st.subheader('SBM Loadings')

  train_df = combined_df.query('is_train==True')
  test_df = combined_df.query('is_train==False')

  f, axes = plt.subplots(ncols=2, figsize=(5, 5))
  axes[0].set_title('train dataset')
  sns.violinplot(split=True, inner="quart",orient="h", data=train_df[loading_features], ax=axes[0], linewidth=0.5)
  
  axes[1].set_title('test dataset')
  sns.violinplot(split=True, inner="quart",orient="h", data=test_df[loading_features], ax=axes[1], linewidth=0.5)
  f.tight_layout()
  sns.despine(left=True)
  st.pyplot()


if show_targets:
  st.header('Targets Correlations')
  f, ax = plt.subplots(figsize=(10, 10))
  sns.heatmap(combined_df[target_features].corr(), linewidths=.5, annot=True, cmap='Blues', square=True, ax=ax)
  #ax.set(xlabel='Target Features', ylabel='Target Features')
  plt.tight_layout()
  st.pyplot()

st.markdown('<hr>', unsafe_allow_html=True)
#------

feat_choices = combined_df.columns.difference([] if show_targets else target_features)
feat_choices = feat_choices.difference([] if show_fnc else fnc_features)
feat_choices = feat_choices.difference([] if show_sbm else loading_features)

feat1 = st.sidebar.selectbox('Select a feature to summerize', feat_choices)

if feat1:

  feat2 = st.sidebar.selectbox('Select a second feature', feat_choices.difference([feat1]))

  summarize_feature([feat1] + ([feat2] if feat2 else []))

  if feat2:

    st.subheader(f'`{feat1}`-`{feat2}` Correlation')
    sns.jointplot(feat1, feat2, data=combined_df, kind="reg", joint_kws = {'scatter_kws': {"alpha": 0.01}})
    st.pyplot()
