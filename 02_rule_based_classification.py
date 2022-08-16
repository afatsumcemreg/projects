###########################################
# Lead Calculation with Rule-Based Classification
###########################################

################# Before Application #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Application #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

# Importing libraries
import pandas as pd
from helpers.eda import *
pd.set_option("display.max_rows", None)

# Task 1: Read the persona.csv file and show the general information about the dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/persona.csv')
df.head()
check_df(df)       # General picture

# Task 2: How many unique source are there? What are their frequencies?
df['source'].value_counts()
df.source.nunique()

# Task 3: How many unique PRICEs are there?
df.price.nunique()

# Task 4: How many sales were made from which PRICE?
df.price.value_counts()

# Task 5: How many sales from which country?
df.country.value_counts()
df.groupby('country')['price'].count()

# Task 6: How much was earned in total from sales by country?
df.groupby('country')['price'].sum()
df.groupby('country').agg({'price': 'sum'})

# Task 7: What are the sales numbers according to SOURCE types?
df.source.value_counts()

# Task 8: What are the PRICE averages by country?
df.groupby('country').agg({'price': 'mean'})

# Task 9: What are the PRICE averages according to SOURCEs?
df.groupby('source').agg({'price': 'mean'})

# Task 10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(['country', 'source']).agg({'price': 'mean'})

# Task 11: What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?
df.groupby(['country', 'source', 'sex', 'age']).agg({'price': 'mean'})

# Task 12: Sort the output by PRICE
agg_df = df.groupby(['country', 'source', 'sex', 'age']).agg({'price': 'mean'}).sort_values('age')
agg_df.head(15)

# Task 13: Convert the names in the index to variable names.
agg_df = agg_df.reset_index()
agg_df.head()

# Task 14: Convert AGE variable to categorical variable and add it to agg_df.
agg_df['age_cat'] = pd.cut(agg_df['age'], bins=[0, 18, 23, 30, 40, agg_df['age'].max()],
                           labels=['0_18', '19_23', '24_30', '31_40', '40_' + str(agg_df['age'].max())])
agg_df.tail()

# Task 15: Define new level based customers and add them as variables to the dataset.
agg_df['customer_level_based'] = ['_'.join(i).upper() for i in agg_df.drop(['age', 'price'], axis=1).values]
agg_df.head()
agg_df = agg_df[['customer_level_based', 'price']]
agg_df.head()

for i in agg_df['customer_level_based'].values:
    print(i.split('_'))

agg_df['customer_level_based'].values

agg_df = agg_df.groupby('customer_level_based').agg({'price': 'mean'})
agg_df = agg_df.reset_index()
agg_df.head()

# Task 16: Segment new customers (USA_ANDROID_MALE_0_18)
agg_df['segment'] = pd.qcut(agg_df['price'], 4, labels=['D', 'C', 'B', 'A'])
agg_df.head(30)
agg_df.groupby('segment')['price'].mean()

# Task 17: Classify the new customers and estimate how much income they can bring
## What segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to earn on average?
new_user = 'TUR_ANDROID_FEMALE_31_40'
agg_df[agg_df['customer_level_based'] == new_user]

## In which segment and on average how much income would a 35-year-old French woman using iOS expect to earn?
new_user = 'FRA_IOS_FEMALE_31_40'
agg_df[agg_df['customer_level_based'] == new_user]

## What segment does a 24-year-old American men using ANDROID belong to and how much income is expected to earn on average?
new_user = 'USA_ANDROID_MALE_24_30'
agg_df[agg_df['customer_level_based'] == new_user]