import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.impute import SimpleImputer

pd.set_option("display.max_columns", 50)

data_df = pd.read_csv("risk-train.txt", sep = '\t')
data_df

data_df.replace('?', pd.NA, inplace = True)

data_df.CLASS.unique()

data_df.head()

null_percentage = data_df.isnull().mean() * 100
null_percentage_table = null_percentage.reset_index()
null_percentage_table.columns = ['Column', 'Null_Percentage']
null_percentage_table

# Removing Null Columns

column_to_drop = null_percentage_table[null_percentage_table['Null_Percentage'] > 20]['Column'].tolist()
data_df.drop(columns = column_to_drop, inplace = True)

# Formatting Columns

data_df["Z_CARD_VALID"] = data_df["Z_CARD_VALID"].astype(str)
data_df[['MONTH', 'YEAR']] = data_df["Z_CARD_VALID"].astype(str).str.split('.', expand = True)

data_df["Z_CARD_VALID_NEW"] = data_df.apply(lambda x: pd.to_datetime(f'{x.YEAR}-{x.MONTH}-01') + pd.offsets.MonthEnd(0), axis = 1)

data_df["Z_CARD_VALID"] = data_df["Z_CARD_VALID_NEW"]
data_df.drop(columns = ['MONTH', 'YEAR', 'Z_CARD_VALID_NEW'], inplace = True)

data_df["TIME_ORDER"] = pd.to_datetime(data_df["TIME_ORDER"], format = '%H:%M', errors = 'coerce')
data_df["TIME_ORDER"] = data_df["TIME_ORDER"].dt.strftime('%H:%M')

data_df["B_BIRTHDATE"] = pd.to_datetime(data_df["B_BIRTHDATE"], errors = 'coerce')

data_df['VALUE_ORDER'] = data_df['VALUE_ORDER'].astype(float)

data_df['AMOUNT_ORDER'] = data_df['AMOUNT_ORDER'].astype(int)

data_df['SESSION_TIME'] = data_df['SESSION_TIME'].astype(int)

data_df['AMOUNT_ORDER_PRE'] = data_df['AMOUNT_ORDER_PRE'].astype(int)

data_df['VALUE_ORDER_PRE'] = data_df['VALUE_ORDER_PRE'].astype(float)

print(data_df.info())
data_df

for column in data_df.columns:
    unique_values = data_df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Creating Columns and updating columns

data_df['AGE'] = data_df['B_BIRTHDATE'].apply(lambda x: (datetime.now() - x).days // 365)
data_df = data_df.drop(columns=['B_BIRTHDATE']) 
data_df.replace('no', 0, inplace = True)

data_df.replace('yes', 1, inplace = True)


data_df = pd.get_dummies(data_df, columns=['Z_METHODE', 'WEEKDAY_ORDER'], drop_first=True)


data_df['TIME_ORDER'] = pd.to_datetime(data_df['TIME_ORDER'], format='%H:%M')
data_df['TIME_ORDER'] = data_df['TIME_ORDER'].apply(lambda x: x.hour * 60 + x.minute)

data_df['Z_CARD_VALID'] = pd.to_datetime(data_df['Z_CARD_VALID'], format='%Y-%m-%d')


data_df['DAYS_TO_CARD_VALID'] = (data_df['Z_CARD_VALID'] - datetime.now()).dt.days


data_df.drop(columns=['Z_CARD_VALID'], inplace=True)

data_df

data_df.CLASS.unique()

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

X = data_df.drop(columns=['CLASS','ORDER_ID'])
y = data_df['CLASS']
imputer = SimpleImputer(strategy='mean')  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6,stratify=y)
print(y.value_counts())
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:\n", conf_matrix)


report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=1)

print("\nClassification Report:\n", report)


import sklearn.metrics as met

met.accuracy_score(y_test,y_pred)

met.precision_score(y_test,y_pred,zero_division=True)



