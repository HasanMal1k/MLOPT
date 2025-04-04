import polars as pl
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
import re
import numpy as np




missing_values=['na',
                'n/a',
                'N/A', 
                'NAN',
                'NA',
                'Null',
                'null',
                'NULL', 
                'Nan', 
                'nan'
                'Unknown', 
                'unknown',
                'UNKNOWN',
                '-',
                '--',
                '---',
                '----',
                '',
                ' ',
                '  ',
                '   ',
                '    ',
                '?',
                '??',
                '???', 
                '????'
                'Missing',
                'missing', 
                'MISSING' ]

df= pl.read_csv('Superstore.csv', null_values=missing_values,ignore_errors=True)
df=df.to_pandas()


# removes white space
df.replace(r'^\s*$', np.nan, regex=True, inplace=True) 
# Drop rows with all null values
df.dropna(how='all', inplace=True)

# Drop columns with more than 95% Null values
to_drop= [column for column in df.columns if (('duplicated') in column or column=='') and df[column].isna().sum()>(df.shape[0]*0.95)]
df.drop(to_drop, axis=1, inplace=True)

#  Nominal Columns for Imputation

df2= df.copy()





nominal_columns = [column for column in df2.columns if df2[column].dtype in ['object', 'category', 'string', 'bool'] ]
non_nominal_columns= [column for column in df2.columns if column not in nominal_columns and df[column].dtype!='datetime64[ns]']
#all_columns= nominal_columns+non_nominal_columns


class ModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes_ = X.mode().iloc[0]  
        return self

    def transform(self, X):
        return X.fillna(self.modes_) 




nominal_imputer=ModeImputer()
non_nominal_imputer= KNNImputer()

preprocessor = ColumnTransformer(
    transformers=[
        ('initial_encoding', nominal_imputer, nominal_columns),
        ('imputing', non_nominal_imputer, non_nominal_columns)
    ],
    remainder='passthrough'  # Keeps unaffected columns unchanged
)


df2=preprocessor.fit_transform(df2)
df2_imputed=pd.DataFrame(df2, columns=nominal_columns+non_nominal_columns)
joblib.dump(preprocessor, 'Cleaning_pipeline.joblib')





#####################   DATE TIME   ########################


# Before encoding for KNNimputation, seperating Date time columns

# If more than 10 entries in a dataset have 1970-01-01, don't make it date time


date_patterns = [
    r"\b\d{4}-\d{2}-\d{2}\b",      # YYYY-MM-DD
    r"\b\d{2}/\d{2}/\d{4}\b",      # MM/DD/YYYY
    r"\b\d{2}-\d{2}-\d{4}\b",      # DD-MM-YYYY
    r"\b\d{2}\.\d{2}\.\d{4}\b",    # MM.DD.YYYY
    r"\b\d{2}\.\d{2}\.\d{2}\b",    # DD.MM.YY
    r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",  # DD-MMM-YYYY
    r"\b[A-Za-z]+\s\d{1,2},\s\d{4}\b", # Month DD, YYYY
    r"\b\d{1,2}/\d{4}\b",          # MM/YYYY
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?\.+\b", # YYYY-MM-DD + any time format (HH:MM:SS etc)
]


def detect_dates(column, patterns):

    for pattern in patterns:
        
        if isinstance(df2[column][1], str) and re.search(pattern, df[column][1]):
            return True

    return False
                
    
date_like=[]
for column in df2_imputed.columns:
    if column not in non_nominal_columns:
        if(detect_dates(column, date_patterns)):
            date_like.append(column)

date_containing = [column for column in date_like if "date" in column.lower()]

for column in date_containing:
    df[column]=pd.to_datetime(df[column], errors='ignore')



df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])