################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

import joblib
import pandas as pd
from helpers.diabetes_pipeline import diabetes_data_prep

df = pd.read_csv("01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv")
X, y = diabetes_data_prep(df)
random_user = X.sample(1, random_state=45)
new_model = joblib.load("voting_clf.pkl")
new_model.predict(random_user)
