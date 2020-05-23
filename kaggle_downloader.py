#%%
# download kaggle dataset
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


competition = ''

for typ in ['fMRI_test', 'fMRI_train']:
  for s in range(10011, 10100):
    f = f'{typ}/{s}.mat'
    print(f'downloading {f}...')
    try:
      api.competition_download_file('trends-assessment-prediction',f, path=f'./data/{typ}/')
      print(f'downloaded {f}...')
    except Exception as e:
      print(f'ignored {f}')
      pass


#%%
import numpy as np

x = np.arange(5)
y = x[::-1]
print(x,y)
