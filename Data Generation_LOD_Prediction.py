#!/usr/bin/env python
# coding: utf-8

# # Data generation 

# In[13]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Data creation
# Generate random sample data for the variables
np.random.seed(50)  # For reproducibility

num_samples = 100000
LOD_raw_husk = np.random.normal(9.6, 0.87, size=num_samples)
swell = np.random.normal(29.87, 2.19, size=num_samples)
on_30 = np.random.normal(42.2, 6.73, size=num_samples)
thru_70 = np.random.uniform(low=0.10, high=6.58, size=num_samples)
thru_100 = np.random.uniform(low=0.5, high=2.5, size=num_samples)

air_velocity = np.random.uniform(low=29, high=32, size=num_samples)
steam_inlet_temp = np.random.uniform(low=147, high=153, size=num_samples)
steam_outlet_temp = np.random.uniform(low=136, high=145, size=num_samples)
steam_sep_line_temp = np.random.uniform(low=115, high=125, size=num_samples)
transfer_cool_air_temp = np.random.uniform(low=63, high=125, size=num_samples)
steam_pressure = np.random.uniform(low=2.6, high=2.8, size=num_samples)
ambient_temp = np.random.uniform(low=20, high=40, size=num_samples)
ambient_humidity = np.random.uniform(low=15, high=55, size=num_samples)
feed_rate = np.random.uniform(low=1000, high=2000, size=num_samples)
moisture_content = -0.00001*air_velocity**2 - 0.07*steam_inlet_temp - 0.06*steam_outlet_temp - 0.078*steam_sep_line_temp
-0.3*transfer_cool_air_temp - 0.004*steam_pressure**4 + 0.09*ambient_temp + 0.3*ambient_humidity+ 0.007*feed_rate**(1/2) + 10*LOD_raw_husk**5 + 0.1*swell**2+0.002*on_30+0.0005*thru_70+0.00001*thru_100 + np.random.normal(0, 0.5 ,size = num_samples)
# np.random.uniform(low=4.5, high=6.5, size=num_samples)

# Rescale the moisture values to the desired range
moisture_content = 4.5 + (moisture_content - np.min(moisture_content))*(6.5 - 4.5) / (np.max(moisture_content) - np.min(moisture_content))

# Combine the variables into a dataframe
data = pd.DataFrame({
    'Air Velocity': air_velocity,
    'Steam Inlet Temp': steam_inlet_temp,
    'Steam Outlet Temp': steam_outlet_temp,
    'Steam Sep Line Temp': steam_sep_line_temp,
    'Transfer cool air temp':transfer_cool_air_temp,
    'Steam Pressure': steam_pressure,
    'Ambient Temp': ambient_temp,
    'Ambient Humidity':ambient_humidity,
    'Feed_rate':feed_rate,
    'LOD_raw':LOD_raw_husk,
    'Swell_vol':swell,
    'on_30':on_30,
    'thru_70':thru_70,
    'thru_100':thru_100,
    'Moisture Content': moisture_content
})
data.to_csv(r'C:\Users\Harshita.Saxena\Downloads\Meta_San_Simulator\sample_data_copy.csv')


# In[14]:


len(data.columns)


# In[15]:


def load_data(filename):
    """
    Load the data from a CSV file into a pandas DataFrame.
    """
    data = pd.read_csv(filename)
    return data

def preprocess_data(data):
    """
    Preprocess the data, including handling missing values, outliers, and transforming variables.
    """
    # Example code for handling missing values
    # data = data.dropna()

    # Example code for log transformation
    # data['column'] = np.log(data['column'])

    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    """
    Standardize/Normalize the data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def build_model(X_train, y_train):
    """
    Build a linear regression model and train it on the training data.

    """
    model = LinearRegression()
    model.fit(X_train, y_train)
     #Get the coefficients and intercept of the linear regression model
    coefficients = model.coef_
    intercept = model.intercept_
    # Print the predicted equation
    equation = "Moisture Content = ({:.5f} + ({:.5f} * Air Velocity) + ({:.5f} * Steam Inlet Temp)+ ({:.5f} * Steam Outlet Temp) + ({:.5f} * Steam Sep Line Temp) + ({:.5f} * Transfer Cool Air Temp)+({:.5f} * Steam Pressure)  +({:.5f} * Ambient Temp)+({:.5f} * Ambient Humidity)+({:.5f} * Feed Rate)+({:.5f} * LOD_raw)+({:.5f} * Swell_vol)+({:.5f} * on_30)+({:.5f} * thru_70)+({:.5f} * thru_100))".format(
        intercept, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4] , coefficients[5],coefficients[6],coefficients[7],coefficients[8], coefficients[9], coefficients[10], coefficients[11], coefficients[12],coefficients[13])
    print("Predicted Equation:\n", equation)

    return model , coefficients , intercept

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on the training and testing data.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    #print(model.summary())
    return train_mse, test_mse, train_r2, test_r2

def make_predictions(model, scaler, data):
    """
    Use the trained model to make predictions on new/unseen data.
    """
    new_data_scaled = scaler.transform(data)
    predictions = model.predict(new_data_scaled)

    return predictions


# In[16]:


# Step 1: Define the problem and load the data
data = load_data(r'C:\Users\Harshita.Saxena\Downloads\Meta_San_Simulator\sample_data_copy.csv')
data= data.drop(['Unnamed: 0'],axis=1)
# Step 2: Explore the data
# Perform EDA and check the structure, missing values, outliers, etc.

# Step 3: Preprocess the data
data = preprocess_data(data)
print(data)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(data, 'Moisture Content')

# Step 5: Standardize/Normalize the data
#X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)

# Step 6: Build the linear regression model
model , coefficients , intercept = build_model(X_train, y_train)

# Step 7: Model evaluation
train_mse, test_mse, train_r2, test_r2 = evaluate_model(model, X_train, y_train, X_test, y_test)


# In[17]:


#Upper and lower bounds
ub_lb_df = pd.DataFrame( { 'Controllable_Parameter' :['Steam Inlet Temp' ,'Steam Outlet Temp','Steam Sep Line Temp','Transfer cool air temp'],
                          'lower' : [147 , 136 , 115 ,63 ] , 
                          'upper': [153 , 145 , 125 , 125] })


# In[18]:


import pickle
model_dict = {'model': model,'coeffecients':coefficients,'intercept':intercept, 'ub_lb_df':ub_lb_df}#caler':scaler}
model_dict

model_file =r"LOD_prediction_training.pkl"
with open(model_file, 'wb') as file:  
    pickle.dump(model_dict, file)


# In[19]:


model_dict


# In[ ]:




