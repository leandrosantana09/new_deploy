import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    
    '''Classe responsavel por add atributos'''
            
    def __init__(self, add_badrooms_per_room=True):
        
        self.add_badrooms_per_room = add_badrooms_per_room
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        
        
        room_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_badrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, room_per_household, population_per_household]
        
        
def full_pipeline(housing):
        
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())]) 
                
    housing_num = housing.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
        ])

    data = full_pipeline.fit(housing)
    
    return data