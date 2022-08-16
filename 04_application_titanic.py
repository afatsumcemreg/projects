# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Import the dataset for the small-scale applications
def load():
    data = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/titanic.csv')
    data.columns = [col.lower() for col in data.columns]
    return data


df = load()
df.head()

# Step 1: Feature engineering

## new_cabin_bool
df['new_cabin_bool'] = df['cabin'].notnull().astype(int)

## name_count
df['new_name_count'] = df['name'].str.len()

## name_word_count
df['new_name_word_count'] = df['name'].apply(lambda x: len(str(x).split(" ")))

## name_dr title
df['new_name_dr'] = df['name'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr.')]))

## name_title
df['new_title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

## family_size
df['new_family_size'] = df['sibsp'] + df['parch'] + 1

## age_pclass
df['new_age_pclass'] = df['age'] * df['pclass']

## is_alone
df.loc[((df['sibsp'] + df['parch']) > 0), 'new_is_alone'] = 'no'
df.loc[((df['sibsp'] + df['parch']) == 0), 'new_is_alone'] = 'yes'

## age_cat
df.loc[(df['age'] < 18), 'new_age_cat'] = 'young'
df.loc[((df['age'] >= 18) & (df['age'] < 56)), 'new_age_cat'] = 'mature'
df.loc[(df['age'] >= 56), 'new_age_cat'] = 'senior'

## new_sex_cat
df.loc[(df['sex'] == 'male') & (df['age'] <= 21), 'new_sex_cat'] = 'young_male'
df.loc[(df['sex'] == 'female') & (df['age'] <= 21), 'new_sex_cat'] = 'young_female'
df.loc[(df['sex'] == 'male') & ((df['age'] > 21) & (df['age'] <= 50)), 'new_sex_cat'] = 'matur_male'
df.loc[(df['sex'] == 'female') & ((df['age'] > 21) & (df['age'] <= 50)), 'new_sex_cat'] = 'matur_female'
df.loc[(df['sex'] == 'male') & (df['age'] > 50), 'new_sex_cat'] = 'senior_male'
df.loc[(df['sex'] == 'female') & (df['age'] > 50), 'new_sex_cat'] = 'senior_female'

df.head()
df.shape


# Step 2: Outliers
## grabing variable names
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    """
    # categorical columns
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['int64', 'float64']]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerical columns
    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Reporting section
    print(f'Observation: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'Number of categorical variables: {len(cat_cols)}')
    print(f'Number of numerical variables: {len(num_cols)}')
    print(f'Number of categorical but cardinal variables: {len(cat_but_car)}')
    print(f'Number of numeric but categorical variables: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in 'passengerid']


# outlier thresholds
def outlier_threshholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    num_cols = [col for col in df.columns if df[col].dtypes in ['int64', 'float64'] and df[col].nunique() > 10]
    num_cols = [col for col in num_cols if col not in 'passengerid']
    for col in num_cols:
        low_limit, up_limit = outlier_threshholds(df, col)
        print(low_limit, up_limit)
    """
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range

    return low_limit, up_limit


## check outliers
def check_outliers(dataframe, col_name):
    """
    for col in num_cols:
        print(col, check_outliers(df, col))
    """
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outliers(df, col))


## replace with thresholds of the outliers
def replace_with_thresholds(dataframe, col_name):
    """
    for col in num_cols:
        replace_with_thresholds(df, col)
    for col in num_cols:
        print(col, check_outliers(df, col))
    """
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outliers(df, col))


# Step 3: Missing values

## missig_values_table
def missing_values_table(dataframe, na_name=False):
    """
    missing_values_table(df)
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 1)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')

    if na_name:
        return na_columns


missing_values_table(df)

## removing 'cabin' variable
df.drop('cabin', inplace=True, axis=1)

## removing 'ticket', and 'name' variables
removed_cols = ['ticket', 'name']
df.drop(removed_cols, inplace=True, axis=1)

## filling with 'median' the misiing values of the variable 'age'
df['age'] = df['age'].fillna(df.groupby('new_title')['age'].transform('median'))

### re-operate the variables depending on the variable 'age'
df['new_age_pclass'] = df['age'] * df['pclass']
df.loc[((df['sibsp'] + df['parch']) > 0), 'new_is_alone'] = 'no'
df.loc[((df['sibsp'] + df['parch']) == 0), 'new_is_alone'] = 'yes'
df.loc[(df['age'] < 18), 'new_age_cat'] = 'young'
df.loc[((df['age'] >= 18) & (df['age'] < 56)), 'new_age_cat'] = 'mature'
df.loc[(df['age'] >= 56), 'new_age_cat'] = 'senior'

## filling with 'mode' the misiing values of the variable 'embarked'
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0)


# Step 4: Label encoding

## label_encoder
def label_encoder(dataframe, binary_col):
    """
    binary_cols = [col for col in df.columns if df[col].dtypes not in ['int64', 'float64'] and df[col].nunique() == 2]
    for col in binary_cols:
        label_encoder(df, col)
    df.sex.head()
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes not in ['int64', 'float64'] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)


# Step 5: Rare analysis

## rare_analyzer
def rare_analyzer(dataframe, target, cat_cols):
    """
    rare_analyzer(dff, 'target', cat_cols)
    """
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            'Count': dataframe[col].value_counts(),
            'Ratio': dataframe[col].value_counts() / len(dataframe),
            'Target_Mean': dataframe.groupby(col)[target].mean()
        }), end='\n\n')


rare_analyzer(df, 'survived', cat_cols)


## rare_encoder
def rare_encoder(dataframe, rare_percent):
    """
    new_df = rare_encoder(dff, 0.01)
    rare_analyzer(new_df, 'target', cat_cols)
    """
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' and
                    (temp_df[col].value_counts() / len(temp_df) < rare_percent).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_percent].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01)
df['new_title'].value_counts()


# Step 6: One Hot Encoding

## one_hot_encoder
def one_hot_encoder(dataframe, categorical_cols, dropfirst=True):
    """
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    one_hot_encoder(df, ohe_cols).head()
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=dropfirst)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

## calling grab_col_names function
cat_cols, num_cols, cat_but_car = grab_col_names(df)

## rare anylsis for cat_cols
rare_analyzer(df, 'survived', cat_cols)

## useless columns
useless_cols = [col for col in df.columns if
                df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# drop useless columns
df.drop(useless_cols, inplace=True, axis=1)
df.head()
df.shape

# Step 7: Standardization

scaler = StandardScaler()
num_cols = [col for col in num_cols if col not in 'passengerid']
df.drop('passengerid', inplace=True, axis=1)
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

# Step 8: Modeling

## Base model

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=['sex', 'embarked'], drop_first=True)
y = dff['survived']
X = dff.drop(['passengerid', 'cabin', 'survived', 'name', 'ticket'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
rf_model_base = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred1 = rf_model_base.predict(X_test)
y_pred2 = rf_model_base.predict(X_train)
accuracy_score(y_pred1, y_test)
accuracy_score(y_pred2, y_train)


## plot_importance of base model
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(rf_model_base, X_train)
plot_importance(rf_model_base, X_test)

## Random Forest Modeling with new variables

## Selecting independent and dependent variables
y = df['survived']
X = df.drop('survived', axis=1)

## train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

## model object
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

## prediction
y_pred1 = rf_model.predict(X_test)
y_pred2 = rf_model.predict(X_train)

## accuracy score
accuracy_score(y_pred1, y_test)
accuracy_score(y_pred2, y_train)


# Step 9: Importance of the new variables
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(rf_model, X_train)
plot_importance(rf_model, X_test)


def all_models(X, y, test_size=0.20, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            precision_train = precision_score(y_train, y_pred_train)
            precision_test = precision_score(y_test, y_pred_test)
            recall_train = recall_score(y_train, y_pred_train)
            recall_test = recall_score(y_test, y_pred_test)
            f1_train = f1_score(y_train, y_pred_train)
            f1_test = f1_score(y_test, y_pred_test)
            roc_auc_train = roc_auc_score(y_train, y_pred_train)
            roc_auc_test = roc_auc_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test, precision_train=precision_train, precision_test=precision_test, recall_train=recall_train, recall_test=recall_test, f1_train=f1_train, f1_test=f1_test, roc_auc_train=roc_auc_train, roc_auc_test=roc_auc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X, y, test_size=0.2, random_state=46, classification=True)


##########################
# CATBOOST MODEL TUNING
##########################
from catboost import CatBoostClassifier
catboost_model = CatBoostClassifier(verbose=False, random_state=42).fit(X_train, y_train)

catboost_params = {"iterations": [200, 400, 600],
                   "learning_rate": [0.01, 0.05, 1.0],
                   "depth": [3, 6, 9]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=10, n_jobs=-1).fit(X_train , y_train)
catboost_best_grid.best_params_

best_params = {"iterations": 600,
               "learning_rate": 0.01,
               "depth": 6}

# CATBOOST TUNED MODEL
catboost_final_model = CatBoostClassifier(iterations=600, learning_rate=0.01, depth=6, random_state=42, verbose=True).fit(X_train, y_train)

# CATBOOST TUNED MODEL TRAIN ERROR
y_pred = catboost_final_model.predict(X_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("CatBoost Tuned Model Train accuracy score:", round(accuracy_score(y_train, y_pred), 4))
print("CatBoost Tuned Model Train precision score:", round(precision_score(y_train, y_pred), 4))
print("CatBoost Tuned Model Train recall score:", round(recall_score(y_train, y_pred), 4))
print("CatBoost Tuned Model Train f1 score:", round(f1_score(y_train, y_pred), 4))
print("CatBoost Tuned Model Train roc_auc score:", round(roc_auc_score(y_train, y_pred), 4))

# CATBOOST TUNED MODEL TEST ERROR
y_pred = catboost_final_model.predict(X_test)

print("CatBoost Tuned Model Test accuracy score:", round(accuracy_score(y_test, y_pred), 4))
print("CatBoost Tuned Model Test precision score:", round(precision_score(y_test, y_pred), 4))
print("CatBoost Tuned Model Test recall score:", round(recall_score(y_test, y_pred), 4))
print("CatBoost Tuned Model Test f1 score:", round(f1_score(y_test, y_pred), 4))
print("CatBoost Tuned Model Test roc_auc score:", round(roc_auc_score(y_test, y_pred), 4))


# CATBOOST TUNED MODEL VISUALIZATION
def plot_importance(model, features, num=len(X), save=False):
    """
    plot_importance(rf_model, X_train)
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(catboost_final_model, X_train)


##########################
# XGBOOST MODEL TUNING
##########################
xgboost_params = {'n_estimators': [100, 500, 1000],
                  'learning_rate': [0.01, 0.05, 1.0],
                  'max_depth': [3, 6, 9],
                  'colsample_bytree': [0.7, 1.0]}

from xgboost import XGBClassifier
xgboost_model = XGBClassifier(random_state=42)

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=10, n_jobs=-1, verbose=True).fit(X_train, y_train)

# XGBOOST TUNED MODEL
xgboost_final_model = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=42).fit(X_train, y_train)

# XGBOOST TUNED MODEL TRAIN ERROR
y_pred = xgboost_final_model.predict(X_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("XGBoost Tuned Model Train accuracy score:", round(accuracy_score(y_train, y_pred), 4))
print("XGBoost Tuned Model Train precision score:", round(precision_score(y_train, y_pred), 4))
print("XGBoost Tuned Model Train recall score:", round(recall_score(y_train, y_pred), 4))
print("XGBoost Tuned Model Train f1 score:", round(f1_score(y_train, y_pred), 4))
print("XGBoost Tuned Model Train roc_auc score:", round(roc_auc_score(y_train, y_pred), 4))

# XGBOOST TUNED MODEL TEST ERROR
y_pred = xgboost_final_model.predict(X_test)

print("XGBoost Tuned Model Test accuracy score:", round(accuracy_score(y_test, y_pred), 4))
print("XGBoost Tuned Model Test precision score:", round(precision_score(y_test, y_pred), 4))
print("XGBoost Tuned Model Test recall score:", round(recall_score(y_test, y_pred), 4))
print("XGBoost Tuned Model Test f1 score:", round(f1_score(y_test, y_pred), 4))
print("XGBoost Tuned Model Test roc_auc score:", round(roc_auc_score(y_test, y_pred), 4))


# XGBOOST TUNED MODEL VISUALIZATION
def plot_importance(model, features, num=len(X), save=False):
    """
    plot_importance(rf_model, X_train)
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(xgboost_final_model, X_train)


##########################
# LIGHTGBM MODEL TUNING
##########################
lgbm_params = {'n_estimators': [100, 300, 500, 1000],
               'learning_rate': [0.01, 0.05, 1.0],
               'colsample_bytree': [0.5, 0.7, 1.0]}

from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(random_state=42)

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X_train, y_train)

# LIGHTGBM TUNED MODEL
lgbm_final_model = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=42).fit(X_train, y_train)

# LIGHTGBM TUNED MODEL TRAIN ERROR
y_pred = lgbm_final_model.predict(X_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("LightGBM Tuned Model Train accuracy score:", round(accuracy_score(y_train, y_pred), 4))
print("LightGBM Tuned Model Train precision score:", round(precision_score(y_train, y_pred), 4))
print("LightGBM Tuned Model Train recall score:", round(recall_score(y_train, y_pred), 4))
print("LightGBM Tuned Model Train f1 score:", round(f1_score(y_train, y_pred), 4))
print("LightGBM Tuned Model Train roc_auc score:", round(roc_auc_score(y_train, y_pred), 4))

# LIGHTGBM TUNED MODEL TEST ERROR
y_pred = lgbm_final_model.predict(X_test)

print("LightGBM Tuned Model Test accuracy score:", round(accuracy_score(y_test, y_pred), 4))
print("LightGBM Tuned Model Test precision score:", round(precision_score(y_test, y_pred), 4))
print("LightGBM Tuned Model Test recall score:", round(recall_score(y_test, y_pred), 4))
print("LightGBM Tuned Model Test f1 score:", round(f1_score(y_test, y_pred), 4))
print("LightGBM Tuned Model Test roc_auc score:", round(roc_auc_score(y_test, y_pred), 4))


# LIGHTGBM TUNED MODEL VISUALIZATION
def plot_importance(model, features, num=len(X), save=False):
    """
    plot_importance(rf_model, X_train)
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(lgbm_final_model, X_train)


##########################
# EXAMNING MODEL COMPLEXITY USING LEARNING CURVES
##########################