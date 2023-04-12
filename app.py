import joblib
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from info_apoio import CombinedAttributesAdder

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
 

    '''Classe responsavel por add atributos'''
            
    def __init__(self, add_badrooms_per_room=True):
        
        self.add_badrooms_per_room = add_badrooms_per_room
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        
        room_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_badrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, room_per_household, population_per_household]

# load model
model = joblib.load(filename='final_model.pkl')
pipeline = joblib.load(filename='pipeline.pkl')

# instanciate flask
app = Flask(__name__)

# route to display the home page
@app.route('/boot') 
def home():
    return render_template("base_boot.html")

# route to display the home page
@app.route('/', methods=['GET']) 
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            longitude = float(request.form['longitude'])
            latitude = float(request.form['longitude'])
            housing_median_age = float(request.form['housing_median_age'])
            total_rooms = float(request.form['total_rooms'])
            total_bedrooms = float(request.form['total_bedrooms'])
            population = float(request.form['population'])
            households = float(request.form['households'])
            median_income = float(request.form['median_income'])
            ocean_proximity = request.form['ocean_proximity']
            
            data = {'longitude': longitude,
                    'latitude': latitude,
                    'housing_median_age': housing_median_age,
                    'total_rooms': total_rooms,
                    'total_bedrooms': total_bedrooms,
                    'population': population,
                    'households': households,
                    'median_income': median_income,
                    'ocean_proximity': ocean_proximity
            }
            
            df_row = pd.DataFrame([data])
            
            # data preparation
            data_final = pipeline.transform(df_row)
                 
            # predict
            pred = model.predict(data_final)
            print(pred)
            return render_template('results.html', pred=round(pred[0], 2))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')        


if __name__ == '__main__':
    
    # start flask
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
    
    
    #<input type="submit" value="Predict">