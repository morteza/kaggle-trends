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

outcomes = ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']

outcomes_weights = [0.3,0.175,0.175,0.175,0.175]


# combine all features and outcomes
combined_df = loading_df.merge(fnc_df, on='Id').merge(train_scores_df, on='Id', how='left')

combined_df.fillna(value={'is_train':False}, inplace=True)

#
# preprocess
#

# give less importance to fnc features, since they can easily overfit due to high dim
FNC_SCALE = 1/500
combined_df[fnc_features] *= FNC_SCALE

# ----------------
# 2. describe data
# ----------------

def show_summary(df):
    print(f'shape:', df.shape)
    display(df.head())
    display(df.describe())
    df.info()
    print('columns:', sorted(df.columns.values))

#show_summary(combined_df)
#show_summary(loading)
#show_summary(train_scores)
# Note: (loading) there is no IC_19, IC_23, IC_25, IC_27


# ----------------
# 3. PLS
# ----------------

def create_pls_model(df):

  def find_best_pls(X, y, cv = 5):
    
    #TODO upsample or downsample
    #TODO X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=None)

    steps = [
      ('impute', SimpleImputer(verbose=10)),
      ('scale', StandardScaler()),
      ('pls', PLSRegression())
    ]
    
    #hyperparameters
    hyper_parameters = {
      'pls__n_components': np.arange(1, 15) #np.arange(1, X.shape[1]) # number of PLS components, max is the number of features
    }

    pipeline = Pipeline(steps)

    #TODO custom scorer
    # grid search for hyper parameters with cross-validation (parallel)
    search = GridSearchCV(pipeline, hyper_parameters, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=10)

    search.fit(X, y) # use all of training

    print(search.best_params_)

    score = search.score(X, y)

    print(f"Test score: {search.scoring} = {score}")

    #y_pred = search.predict(X_test)
    #error = np.sqrt(mean_squared_error(y_test, y_pred))

    #plot_precision_recall_curve(search, X_test, y_test)
    #plot_confusion_matrix(search, X_test, y_test)

    return search


  #DEBUG display(loading.Id.nunique(), train_scores.Id.nunique())
  #DEBUG display(df.describe())

  X = df[features].copy()
  y = df[outcomes].copy()

  imputer = SimpleImputer()
  y = pd.DataFrame(imputer.fit_transform(y))
  y.columns = outcomes

  # df = df.dropna(subset=outcomes)
  return find_best_pls(X, y)



train_df = combined_df.query('is_train==True')
test_df =  combined_df.query('~(is_train==True)')

model = create_pls_model(train_df)

# only predict test cohort
y_pred = model.predict(test_df[features])

pred = pd.DataFrame(y_pred)
#pred.columns = outcomes
#pred['Id'] = test_df.Id

test_df[outcomes] = pred # test_df.merge(pred, on='Id', how='left')

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

  submission.to_csv(submissions_dir / 'submission_pls.csv', index=False, columns=['Id','Predicted'])

  return submission[['Id','Predicted']].sort_values("Id")


submission_df = create_submission(test_df, submissions_dir / 'submission_pls.csv')

display(submission_df)

assert submission_df.shape[0] == 29385 # submission rule
