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

# features and prediction targets
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
features = loading_features + fnc_features

targets = ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']

target_weights = [0.3,0.175,0.175,0.175,0.175]


# combine all features and targets
combined_df = loading_df.merge(fnc_df, on='Id').merge(train_scores_df, on='Id', how='left')

combined_df.fillna(value={'is_train':False}, inplace=True)

#
# preprocess
#

# give less importance to fnc features, since they can easily overfit due to high dim
#FNC_SCALE = 1/500
#combined_df[fnc_features] *= FNC_SCALE

train_df = combined_df.query('is_train==True')
test_df =  combined_df.query('~(is_train==True)')

#train_df[targets].describe()

#missing_ratio = train_df.isna().sum() * 100 / len(train_df)

# impute all missing targets with -1!!
train_df = train_df.fillna(-1)

#%%


import autokeras as ak
import tensorflow as tf


# Initialize the multi-input/multi-task model
model_inputs = [ak.StructuredDataInput(), ak.StructuredDataInput()]
model_outputs = [ak.RegressionHead() for t in targets]

print('compiling the model...')
model = ak.AutoModel(
  name='loading_fnc',
  directory='tmp/autokeras',
  inputs=model_inputs,
  outputs=model_outputs,
  max_trials=2)


# Fit the model with prepared data.
print('fitting the model...')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tmp/logs')
model.fit(
  [train_df[loading_features],train_df[fnc_features]],
  [train_df[t] for t in targets],
  epochs=1,
  callbacks = [tensorboard_callback]
  )

print('fitting finished!')

accuracy = model.evaluate(
  [train_df[loading_features], train_df[fnc_features]], 
  [train_df[t] for t in targets]
  )

#DEBUG print('Accuracy:', accuracy)


# save the model
model.export_model().save("models/autokeras_loading_fnc", save_format="tf")

#tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

# predict: only predict test cohort
y_pred = model.predict([test_df[loading_features], test_df[fnc_features]])

pred = pd.DataFrame(np.concatenate(y_pred, axis=1))
#pred.columns = targets
#pred['Id'] = test_df.Id

test_df[targets] = pred # test_df.merge(pred, on='Id', how='left')

# ----------------
# 4. Create submission csv
# ----------------


def create_submission(pred, output_file, targets=targets, sample_submission=sample_submission_df, impute=True):

  submission = pred.melt(id_vars=['Id'],value_name='Predicted', value_vars=targets)
  submission['Id'] = submission['Id'].astype('str') + '_' + submission['variable']

  submission = pd.merge(sample_submission, submission, on='Id', how='outer', suffixes=('_sample', ''))

  if impute:
    imputer = SimpleImputer()
    submission['Predicted'] = pd.DataFrame(imputer.fit_transform(submission[['Predicted']]))

  submission.to_csv(output_file, index=False, columns=['Id','Predicted'])

  return submission[['Id','Predicted']].sort_values("Id")


submission_df = create_submission(test_df, submissions_dir / 'submission_autokeras.csv')

assert submission_df.shape[0] == 29385 # submission rule
