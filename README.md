# Titanic Survival Prediction
> Binary classification project using Python and scikit-learn — Logistic Regression model achieving **73.18% accuracy**.

## Overview
This project builds a machine learning pipeline to predict passenger survival on the Titanic. It covers the full data science workflow: data loading, cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Real Results
| Metric | Score |
|--------|-------|
| Model | Logistic Regression |
| Accuracy | **73.18%** |
| Dataset | 891 training records (Titanic dataset) |
| True Negatives | 94 |
| False Positives | 11 |
| False Negatives | 37 |
| True Positives | 37 |

## Project Workflow
1. **Data Loading** — loaded Titanic CSV dataset using Pandas
2. **Data Cleaning** — filled missing Age values with median, missing Embarked with mode, missing Fare with median
3. **Feature Encoding** — one-hot encoded Embarked column; mapped Sex to binary (male=1, female=0)
4. **Exploratory Data Analysis** — plotted age and fare distributions, survival rates by passenger class, sex, and SibSp; produced correlation heatmap
5. **Model Training** — split data 80/20 train/test; applied StandardScaler; trained Logistic Regression model
6. **Evaluation** — assessed performance using accuracy score and confusion matrix

## Key Findings
- Female passengers had significantly higher survival rates than male passengers
- First-class passengers survived at higher rates than second and third class
- Age was not a strong standalone predictor after handling missing values

## Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook (Anaconda)

## Key Skills Demonstrated
- End-to-end machine learning pipeline
- Handling missing data with imputation strategies
- Feature encoding (one-hot, binary mapping)
- Exploratory data analysis and visualisation
- Logistic regression classification and evaluation (accuracy, confusion matrix)

## How to Run
```bash
git clone https://github.com/yashparmar1/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook titanic_logistic_regression.ipynb
```

## Dataset
Titanic dataset — classic binary classification dataset with 891 passenger records including survival outcome, class, sex, age, fare, and embarkation details.

## Author
**Yashkumar Parmar**
MSc Data Science — Middlesex University London (2026)
[LinkedIn](https://www.linkedin.com/in/yashkumar-parmar-7752b8238) · [GitHub](https://github.com/yashparmar1)


