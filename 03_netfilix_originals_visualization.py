# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import warnings
from helpers.eda import *
from helpers.data_prep import *
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Redaing the dataset
df = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/netflix_originals.csv', encoding='latin-1')
df.columns = [col.lower() for col in df.columns]
df.head()

# Checking the dataframe
check_df(df)

# Grabing variables and their types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Summary of the numerical columns
for col in num_cols:
    num_summary(df, col, plot=True)

# Outlier thresholds
for col in num_cols:
    outlier_thresholds(df, col, q1=0.10, q3=0.90)

# Checking the outliers
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.10, q3=0.90))
for col in num_cols:
    replace_with_thresholds(df, col, q1=0.10, q3=0.90)

# In which language were the long-running films created according to the dataset? Make a visualization
df.groupby('language').agg({'runtime': 'sum'}).sort_values('runtime', ascending=False).head(10).plot(kind='bar')
plt.xlabel('Language')
plt.ylabel('Runtime')
plt.show(block=True)

# Find and visualize the IMDB values of the movies shot in the 'Documentary' genre between January 2019 and June 2020.
df['premiere'] = pd.to_datetime(df['premiere'])
df_premiere = df[(df['genre'] == 'Documentary') & (df['premiere'].between('2019-01-01', '2020-06-30'))].sort_values('imdb score')
df_premiere.head()

plt.figure(figsize=(12, 5))
sns.barplot(y=df_premiere['imdb score'], x=df_premiere['title'], orient='v')
plt.title('IMDB values of the movies shot in the "Documentary" genre between January 2019 and June 2020')
plt.xticks(rotation=90)
plt.show(block=True)

# Which genre has the highest IMDB rating among movies shot in English?
df_english = df[df['language'] == 'English'].sort_values('imdb score', ascending=False)
df_english.head()
sns.barplot(x=df_english['genre'].head(10), y=df_english['imdb score'])
plt.xticks(rotation=90)
plt.show(block=True)

# What is the average 'runtime' of movies shot in "Hindi" language?
df[df['language'] == 'Hindi']['runtime'].mean()

# How many categories does the "Genre" Column have, and what are those categories? Express it visually.
df['genre'].value_counts().plot(kind='bar')
plt.xlabel('Genre')
plt.xticks(rotation=90)
plt.show(block=True)

# Find the 3 most used languages in the movies in the data set.
df_lang = df['language'].value_counts()
df_lang.head(3).plot(kind='bar')
plt.show(block=True)

# What are the top 10 movies with the highest IMDB rating?
df_movie = df[['title', 'imdb score']].sort_values('imdb score', ascending=False).head(10)
sns.barplot(x='title', y='imdb score', data=df_movie)
plt.xticks(rotation=90)
plt.show(block=True)

# What is the correlation between IMDB score and 'Runtime'? Examine and visualize.
sns.regplot(x='imdb score', y='runtime', data=df)
plt.show(block=True)
x = round(df['imdb score'].corr(df['runtime']), 3)
print(f'The correlation between runtime and imdb score is {x}.')

# Which are the top 10 'Genre' with the highest IMDB Score? Visualize it.
df[['genre', 'imdb score']].sort_values('imdb score', ascending=False).drop_duplicates('genre').head(10).plot(x='genre', y='imdb score', kind='bar')
plt.xlabel('Genre')
plt.ylabel('IMDB Score')
plt.show(block=True)

# What are the top 10 movies with the highest 'runtime'? Visualize it.
df[['title', 'runtime']].sort_values('runtime', ascending=False).head(10).plot(x='title', y='runtime', kind='bar')
plt.xlabel('Movie Title')
plt.ylabel('Runtime')
plt.show(block=True)

# In which year was the most movies released? Visualize it.
df['year'] = df['premiere'].dt.year
df.groupby('year').agg({'title': 'count'}).sort_values('title', ascending=False).plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Film count')
plt.show(block=True)

# Which language movies have the lowest average IMBD ratings? Visualize it.
df.groupby('language').agg({'imdb score': 'mean'}).head(10).sort_values('imdb score', ascending=True).plot(kind='bar')
plt.xlabel('Language')
plt.ylabel('IMDB Score')
plt.show(block=True)

# Which year has the greatest total runtime?
df.groupby('year').agg({'runtime': 'sum'}).sort_values('runtime', ascending=True).plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Sum of runtime')
plt.show(block=True)

# What is the "Genre" where each language is used the most?
dff = df.groupby(['language']).agg({'genre': 'max'})
dff.reset_index(inplace=True)
dff.head()

# Is there any outlier data in the data set? Please explain.
num_cols
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['runtime'])
plt.subplot(1, 2, 2)
sns.boxplot(y=df['imdb score'])
plt.show(block=True)