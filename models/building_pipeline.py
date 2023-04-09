# basic imports
import pandas as pd
import numpy as np
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from info_apoio import CombinedAttributesAdder


housing = pd.read_csv("C:\\Users\\leand\\OneDrive\\Documentos\\oreilly_project\\datasets\\housing\\housing.csv")

# setting data
housing = housing.drop('median_house_value', axis=1)

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())])             
                
housing_num = housing.drop('ocean_proximity', axis=1)
                    
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
    ])

data_final = full_pipeline.fit_transform(housing)

print(data_final.shape)

joblib.dump(full_pipeline, 'pipeline.pkl')

print(full_pipeline)