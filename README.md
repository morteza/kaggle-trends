# Kaggle TReNDS

## Prepration

```bash
conda env create -f environment.yml
conda activate kaggle-trends
```

## Submit to kaggle

First make sure you installed and authenticated kaggle cli. Then:

```bash
kaggle competitions submit -c trends-assessment-prediction -f ./data/submission_*.csv -m "first SVM model"
```
