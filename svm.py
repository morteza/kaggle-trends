#%%
from collections import namedtuple

from IPython.display import display

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

submissions_dir = pathlib.Path('./submissions')
data_dir = pathlib.Path('./data')

fnc_df = pd.read_csv(data_dir / 'fnc.csv')
loading_df = pd.read_csv(data_dir / 'loading.csv')
sample_submission_df = pd.read_csv(data_dir / 'sample_submission.csv')

train_scores_df = pd.read_csv(data_dir / 'train_scores.csv')

train_scores_df['is_train'] = True

# features and prediction outcomes
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
features = loading_features #+ fnc_features


# outcome: (weight, regularization)
outcomes_info = {
  'age': (0.3, 100),
  'domain1_var1': (0.175, 10),
  'domain1_var2': (0.175, 10),
  'domain2_var1': (0.175, 10),
  'domain2_var2': (0.175, 10)
}

outcomes = list(outcomes_info.keys())

# combine all features and outcomes
combined_df = loading_df.merge(fnc_df, on='Id').merge(train_scores_df, on='Id', how='left')

combined_df.fillna(value={'is_train':False}, inplace=True)

#
# preprocess
#

# give less importance to fnc features, since they can easily overfit due to high dim
FNC_SCALE = 1/100
combined_df[fnc_features] *= FNC_SCALE

# ----------------
# 3. SVC
# ----------------

def create_svm_model(df, outcome):
    
  X = df[features].copy()
  y = df[outcome].copy()

  imputer = SimpleImputer()
  y = pd.DataFrame(imputer.fit_transform(y))
  y.columns = outcome

  #TODO upsample or downsample
  #TODO X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=None)

  steps = [
    ('impute', SimpleImputer(verbose=10)),
    #('scale', StandardScaler()),
    ('svr', SVR())
  ]
    
  #hyperparameters
  hyper_parameters = {
    'svr__C': [1,2,3,4,5,6,7,8,9,10,15,20,50,80,100]
  }

  pipeline = Pipeline(steps)

  #TODO custom scorer
  # grid search for hyper parameters with cross-validation (parallel)
  search = GridSearchCV(pipeline, hyper_parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=10)

  search.fit(X, y.iloc[:,0]) # use all of training

  print(search.best_params_)

  #y_pred = search.predict(X_test)
  #error = np.sqrt(mean_squared_error(y_test, y_pred))

  #plot_precision_recall_curve(search, X_test, y_test)
  #plot_confusion_matrix(search, X_test, y_test)
  return search


train_df = combined_df.query('is_train == True')
test_df  = combined_df.query('is_train == False')

for o in outcomes:
  model = create_svm_model(train_df, outcome=[o])

  # only predict test cohort
  y_pred = model.predict(test_df[features])

  test_df.loc[:,o] = y_pred

display(test_df[outcomes])

#%%
# ----------------
# 4. Create submission csv
# ----------------


def create_submission(pred, output_file, outcomes=outcomes, sample_submission=sample_submission_df, impute=True):

  submission = pred.melt(id_vars=['Id'],value_name='Predicted', value_vars=outcomes)
  submission['Id'] = submission['Id'].astype('str') + '_' + submission['variable']

  submission = pd.merge(sample_submission, submission, on='Id', how='outer', suffixes=('_sample', ''))

  if impute:
    imputer = SimpleImputer()
    submission['Predicted'] = pd.DataFrame(imputer.fit_transform(submission[['Predicted']]))

  submission.to_csv(submissions_dir / 'submission_svm.csv', index=False, columns=['Id','Predicted'])

  return submission[['Id','Predicted']].sort_values("Id")


submission_df = create_submission(test_df, submissions_dir / 'submission_svm.csv')

display(submission_df)

assert submission_df.shape[0] == 29385 # submission rule
