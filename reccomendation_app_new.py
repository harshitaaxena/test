#!/usr/bin/env python
# coding: utf-8

# In[5]:


##Recommendation App


# In[142]:


import pickle
from PIL import Image
import numpy as np 
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

model_file = r"C:\Users\Harshita.Saxena\Downloads\Meta_San_Simulator\LOD_prediction_training.pkl"
with open(model_file, 'rb') as file:  
        parameters = pickle.load(file)
        


# In[120]:


#parameters


# In[121]:


coefficients = parameters['coeffecients']
intercept= parameters['intercept']
ub_lb_df = parameters['ub_lb_df']
#scaler = parameters['scaler']


# In[122]:


#heuristic logic
def reccomendation_generator(input_data,ub_lb_df,coefficients,intercept,predicted_moisture_content,min_,max_):
    
    '''The function takes as input user entered data for controllable parameters and min and max LOD ranges ,
    predicts LOD levels and recommends parameters to reach target LODs
    Input :
    input_data : DataFrame containing user input of controllable params , external params and fixed params
    coefficients : regression coeffs
    intercept : Regression intercept
    predicted_moisture = LOD prediction given user input
    min_ = Minimum allowed LOD as per user input
    max_ = Maximum allowed LOD given user input
    
    Output :
    reccomendation_df : reccommended parameters value in each iteration'''
    rec = pd.DataFrame()
    input_data_init =input_data.copy()
    input_data_init['Iteration'] = 0
    input_data_init['LOD_achieved'] = predicted_moisture_content[0]

    coefficients_dict = dict(zip(list(input_data.columns),list(np.append(coefficients,[intercept]))))
    # List of keys to include in the subset
    controllable_params = ['Steam Inlet Temp', 'Steam Outlet Temp', 'Steam Sep Line Temp' , 'Transfer cool air temp']

    # Subset the dictionary based on the keys
    controllable_params_dict = {key: coefficients_dict[key] for key in controllable_params if key in coefficients_dict}
    # Sort the dictionary items based on the absolute value of coefficients in increasing order
    sorted_coefficients = sorted(controllable_params_dict.items(), key=lambda x: abs(x[1]),reverse=True)

    # Create a new dictionary with sorted variable names and coefficients
    sorted_coefficients_dict = {k: v for k, v in sorted_coefficients}

    # Convert values to lists
    coefficients_dict = {key: [value] for key, value in coefficients_dict.items()}
    coefficients_df = pd.DataFrame(coefficients_dict)
    if (predicted_moisture_content[0] < min_):
        reqd_moisture_list = list(np.arange(round(predicted_moisture_content[0],2),round(min_+ (max_-min_)/2,2),(max_-min_)/2))
        input_data_init['Step Wise Target LOD Values'] = reqd_moisture_list[0]
    elif (predicted_moisture_content[0] > max_) :
        reqd_moisture_list = list(np.arange(round(min_+(max_-min_)/2,2),round(predicted_moisture_content[0],2),(max_-min_)/2))[::-1]
        input_data_init['Step Wise Target LOD Values'] = reqd_moisture_list[0]
    else :
        reqd_moisture_list = 'None'       
        input_data_init['Step Wise Target LOD Values'] = predicted_moisture_content[0]
    input_data1 = pd.DataFrame()    
    if  reqd_moisture_list!= 'None':
        reccommendation_df = pd.DataFrame()
        i = 0 
        iteration = []
        for y in reqd_moisture_list:
            i += 1 
            reqd_moisture_content = y
            recc = pd.DataFrame()
            var_list = []
            recc_list = []
            lod_acheived = []
            lod_adjusted = []
            iteration = []
            for x in sorted_coefficients_dict.keys():
                iteration = iteration + [i]
                lod_adjusted = lod_adjusted + [reqd_moisture_content]
                input_data = input_data.copy()
                input_data_ = input_data.drop([x],axis=1)
                for a in ['Iteration','Step Wise Target LOD Values','LOD_achieved']:
                    if a in list(input_data_.columns):
                        input_data_ = input_data_.drop(a,axis=1)    
                coefficients_ = coefficients_df.drop(x,axis=1)
                coeff_curr = coefficients_df.loc[0,x]
                adj_value = (reqd_moisture_content - np.dot(np.array(input_data_),np.array(coefficients_)[0]))/coeff_curr
                #print(x)
                #print(adj_value)
                if adj_value <= float(ub_lb_df[ub_lb_df['Controllable_Parameter']==x].lower):
                    adj_value = float(ub_lb_df[ub_lb_df['Controllable_Parameter']==x].lower)
                elif adj_value >= float(ub_lb_df[ub_lb_df['Controllable_Parameter']==x].upper):
                    adj_value = float(ub_lb_df[ub_lb_df['Controllable_Parameter']==x].upper)
                else :
                    adj_value = adj_value[0]
                input_data[x] = adj_value
                #print(input_data)
                input_data_copy = input_data.copy()
                for b in ['Iteration','Step Wise Target LOD Values','LOD_achieved']:
                    if b in list(input_data_copy.columns):
                        input_data_copy= input_data_copy.drop(b,axis=1)
                updated_lod = np.dot(np.array(input_data_copy),np.array(coefficients_df)[0])[0]
                input_data['LOD_achieved'] = updated_lod
                input_data['Step Wise Target LOD Values'] = reqd_moisture_content
                input_data['Iteration']= i
                #print(input_data)
                lod_acheived = lod_acheived + [updated_lod]
                var_list = var_list + [x]
                recc_list = recc_list + [adj_value]
                #print(adj_value)
                df_ = pd.DataFrame({'Iteration':iteration,'Step Wise Target LOD Values': lod_adjusted,
                           'Parameter':var_list, 'Reccomended Value(In Range)': recc_list,'LOD_achieved':lod_acheived})
                #print(df_)
                input_data1 = pd.concat([input_data1,input_data],ignore_index=True)
            reccommendation_df=  pd.concat([reccommendation_df,df_], ignore_index=True)
        input_data_f = pd.concat([input_data_init,input_data1], ignore_index= True)
        print("Maximum Convergence Limit Reached")
        for z in list(input_data_f.Iteration.unique()):
            recc_df_ = input_data_f[input_data_f.Iteration == z].reset_index()
            df_1 = recc_df_.iloc[[-1]]
            rec = pd.concat([rec,df_1],axis=0)
        rec1 = rec[['Iteration','Step Wise Target LOD Values','LOD_achieved','Steam Inlet Temp', 'Steam Outlet Temp','Steam Sep Line Temp', 'Transfer cool air temp']]
    else :
        print("No Recommendation required")
        rec1 = pd.DataFrame()
    return  rec1


# In[140]:


data = pd.read_csv(r'C:\Users\Harshita.Saxena\Downloads\Meta_San_Simulator\sample_data.csv')
data= data[[ 'Air Velocity', 'Steam Inlet Temp', 'Steam Outlet Temp',
       'Steam Sep Line Temp', 'Transfer cool air temp', 'Steam Pressure',
       'Ambient Temp', 'Feed_rate', 'Moisture Content']]
data1 = data[['Steam Inlet Temp', 'Steam Outlet Temp',
       'Steam Sep Line Temp', 'Transfer cool air temp', 'Steam Pressure','Moisture Content']]


# In[141]:



# # Sidebar for variable selection
# st.sidebar.title("Variable Selection")
# dependent_variable = st.sidebar.selectbox("LOD of Sanitized Husk", data.columns)
# independent_variables = st.sidebar.multiselect("Independent Variables", data.columns)


# filtered_data = data[[dependent_variable] + independent_variables]

# # Summary statistics
# st.header("Summary Statistics")
# st.write(filtered_data.describe())

# # Correlation matrix
# st.header("Correlation Matrix")
# corr_matrix = filtered_data.corr()
# st.write(corr_matrix)

# # Pairplot
# st.header("Pairplot")
# sns.pairplot(filtered_data, diag_kind='kde')
# st.pyplot()


# In[127]:


#data = pd.read_csv(r'C:\Users\Harshita.Saxena\Downloads\Meta_San_Simulator\sample_data.csv')

# # Calculate correlation
# correlation = data['Variable1'].corr(data['Variable2'])
# print("Correlation:", correlation)
# Streamlit app

# st.title("Bivariate Analysis")
# for j in list(data1.drop(['Moisture Content'],axis =1).columns.unique()):
#     data_ = data1[[j,'Moisture Content']]
#     correlation = data_[j].corr(data_['Moisture Content'])
#     st.write("Change " + j + " variable using the slider below:")

#     # # Slider for input variable
#     input_value = st.slider(j, min_value=0, max_value=200, value=3, step=1)

#     # # Filter data based on the selected input value
#     filtered_data = data_[data_[j] == input_value]

#     # # Plotting bivariate graph
#     plt.figure(figsize=(4, 3))
#     plt.scatter(filtered_data[j], filtered_data['Moisture Content'])
#     plt.xlabel(j)
#     plt.ylabel('Moisture Content')
#     plt.title('Bivariate Analysis')
#     st.pyplot()


# In[105]:


# Define page names
HOME = "Home"
PAGE_1 = "Sensitivity Analysis"
PAGE_2 = "Prediction for Sanitized Husk LOD and Reccomendation"

# Create navigation buttons
nav_option = st.sidebar.radio("Navigation", (HOME, PAGE_1, PAGE_2))

# Display content based on selected page
if nav_option == HOME:
    st.title("Home Page")
    st.write("Welcome to the Home Page!")
    # Sidebar for variable selection
    st.sidebar.title("Variable Selection")
    dependent_variable = st.sidebar.selectbox("LOD of Sanitized Husk", data.columns)
    independent_variables = st.sidebar.multiselect("Independent Variables", data.columns)


    filtered_data = data[[dependent_variable] + independent_variables]

    # Summary statistics
    st.header("Summary Statistics")
    st.write(filtered_data.describe())

    # Correlation matrix
    st.header("Correlation Matrix")
    corr_matrix = filtered_data.corr()
    st.write(corr_matrix)

    # Pairplot
    st.header("Pairplot")
    sns.pairplot(filtered_data, diag_kind='kde')
    st.pyplot()

elif nav_option == PAGE_1:
    st.title("Sensitivity Analysis")
    #st.title("Bivariate Analysis")
    for j in list(data1.drop(['Moisture Content'],axis =1).columns.unique()):
        data_ = data1[[j,'Moisture Content']]
        correlation = data_[j].corr(data_['Moisture Content'])
        st.write("Change " + j + " variable using the slider below:")

        # # Slider for input variable
        input_value = st.slider(j, min_value=0, max_value=200, value=3, step=1)

        # # Filter data based on the selected input value
        filtered_data = data_[data_[j] == input_value]

        # # Plotting bivariate graph
        plt.figure(figsize=(8,6))
        plt.scatter(filtered_data[j], filtered_data['Moisture Content'])
        plt.xlabel(j)
        # Set the range of the x and y axes
        plt.xlim([0, 200])
        plt.ylim([1, 5])
        plt.ylabel('Moisture Content')
        plt.title('Bivariate Analysis')
        st.pyplot()

elif nav_option == PAGE_2:
    # Create a Streamlit web app
    # Create a Streamlit web app
    st.title("Loss of Drying Prediction")

    # Upload manufacturing image
    image = Image.open(r"C:\Users\Harshita.Saxena\Downloads\manufacturing_img.jpeg")
    st.image(image, use_column_width=True)

    # Display subheading for external temperature
    st.subheader("External Parameters")
    input_ambient_temp = st.number_input("Enter Ambient temp:" , value = 30)

    st.subheader("Controllable Parameters")
    # User input for the controllable independent variables

    input_steam_inlet_temp = st.number_input("Enter Steam Inlet temp: " , value = 153)
    input_steam_outlet_temp = st.number_input("Enter Steam Outlet temp: ",value = 145)
    input_steam_sep_line_temp = st.number_input("Enter Steam sep line temp: ",value = 145)
    input_transfer_cool_air_temp = st.number_input("Enter transfer cool air temp:",value = 63)

    st.subheader('Parameters Not Controllable')
     #Display subheading for fixed input parameters


    # Display the fixed value for steam pressure
    avg_steam_pressure = st.number_input("Enter Steam Pressure: " , value = 2.6)
    avg_feed_rate = st.number_input("Enter Feed Rate: " , value = 1000)
    avg_air_velocity = st.number_input("Enter Air Velocity: " , value = 2.5)

    # Create a dataframe with the user input and fixed values


    input_data_ = pd.DataFrame({
        'Air Velocity': [float(avg_air_velocity)],
        'Steam Inlet Temp': [float(input_steam_inlet_temp)],
        'Steam Outlet Temp': [float(input_steam_outlet_temp)],
        'Steam Sep Line Temp': [float(input_steam_sep_line_temp)],
        'Transfer cool air temp':[float(input_transfer_cool_air_temp)],
        'Steam Pressure': [float(avg_steam_pressure)],
        'Ambient Temp': [float(input_ambient_temp)],
        'Feed_rate':[float(avg_feed_rate)]
     })
    # user_input = np.append(np.array([[float(input_air_velocity),float(input_steam_inlet_temp),float(input_steam_outlet_temp),float(input_steam_sep_line_temp),
    #                      float(input_transfer_cool_air_temp),float(input_steam_pressure),float(input_ambient_temp),float(input_feed_rate)]]),1)
    # # Add a constant term for the intercept
    input_data_['const'] = 1.0 #sm.add_constant(input_data)



    # Predict the moisture content using the trained model


    # Display the predicted moisture content
    st.subheader("Predicted Moisture Content")
    predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
    #Display User input min max LOD values
    input_min_LOD = float(st.number_input("Enter Minimum allowed LOD :", value = 0))
    input_max_LOD = float(st.number_input("Enter Maximum allowed LOD :", value = 1))





    if ((predicted_moisture_content[0] < input_min_LOD)|(predicted_moisture_content[0] > input_max_LOD)):
        st.markdown(
            f'<div style="background-color: red; padding: 10px; border-radius: 5px;">'
            f'<p style="color: white; font-weight: bold;">{predicted_moisture_content}</p>'
            f'<p style="color: white;">Outside Target Range</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color: green; padding: 10px; border-radius: 5px;">'
            f'<p style="font-weight: bold;">{predicted_moisture_content}</p>'
            f'</div>',
#             unsafe_allow_html=True
        )

    # Button for issue input parameter recommendations
    if ((predicted_moisture_content[0] < input_min_LOD)|(predicted_moisture_content[0] > input_max_LOD)):
        recc_df = reccomendation_generator(input_data_,ub_lb_df,coefficients,intercept,predicted_moisture_content,input_min_LOD,input_max_LOD)
        if st.button("Issue Input Parameter Recommendations"):
            st.write("Recommendation: Adjust the input parameters to bring the moisture content within the target range.")
            st.dataframe(recc_df)


# In[8]:


##Stream Lit App


# In[98]:


# # Create a Streamlit web app
# st.title("Loss of Drying Prediction")

# # Upload manufacturing image
# image = Image.open(r"C:\Users\Harshita.Saxena\Downloads\manufacturing_img.jpeg")
# st.image(image, use_column_width=True)

# # Display subheading for external temperature
# st.subheader("External Parameters")
# input_ambient_temp = st.number_input("Enter Ambient temp:" , value = 30)

# st.subheader("Controllable Parameters")
# # User input for the controllable independent variables

# input_steam_inlet_temp = st.number_input("Enter Steam Inlet temp: " , value = 153)
# input_steam_outlet_temp = st.number_input("Enter Steam Outlet temp: ",value = 145)
# input_steam_sep_line_temp = st.number_input("Enter Steam sep line temp: ",value = 145)
# input_transfer_cool_air_temp = st.number_input("Enter transfer cool air temp:",value = 63)

# st.subheader('Parameters Not Controllable')
#  #Display subheading for fixed input parameters


# # Display the fixed value for steam pressure
# avg_steam_pressure = st.number_input("Enter Steam Pressure: " , value = 2.6)
# avg_feed_rate = st.number_input("Enter Feed Rate: " , value = 1000)
# avg_air_velocity = st.number_input("Enter Air Velocity: " , value = 2.5)

# # Create a dataframe with the user input and fixed values


# input_data_ = pd.DataFrame({
#     'Air Velocity': [float(avg_air_velocity)],
#     'Steam Inlet Temp': [float(input_steam_inlet_temp)],
#     'Steam Outlet Temp': [float(input_steam_outlet_temp)],
#     'Steam Sep Line Temp': [float(input_steam_sep_line_temp)],
#     'Transfer cool air temp':[float(input_transfer_cool_air_temp)],
#     'Steam Pressure': [float(avg_steam_pressure)],
#     'Ambient Temp': [float(input_ambient_temp)],
#     'Feed_rate':[float(avg_feed_rate)]
#  })
# # user_input = np.append(np.array([[float(input_air_velocity),float(input_steam_inlet_temp),float(input_steam_outlet_temp),float(input_steam_sep_line_temp),
# #                      float(input_transfer_cool_air_temp),float(input_steam_pressure),float(input_ambient_temp),float(input_feed_rate)]]),1)
# # # Add a constant term for the intercept
# input_data_['const'] = 1.0 #sm.add_constant(input_data)



# # Predict the moisture content using the trained model


# # Display the predicted moisture content
# st.subheader("Predicted Moisture Content")
# predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
# #Display User input min max LOD values
# input_min_LOD = float(st.number_input("Enter Minimum allowed LOD :", value = 0))
# input_max_LOD = float(st.number_input("Enter Maximum allowed LOD :", value = 1))





# if ((predicted_moisture_content[0] < input_min_LOD)|(predicted_moisture_content[0] > input_max_LOD)):
#     st.markdown(
#         f'<div style="background-color: red; padding: 10px; border-radius: 5px;">'
#         f'<p style="color: white; font-weight: bold;">{predicted_moisture_content}</p>'
#         f'<p style="color: white;">Outside Target Range</p>'
#         f'</div>',
#         unsafe_allow_html=True
#     )
# else:
#     st.markdown(
#         f'<div style="background-color: green; padding: 10px; border-radius: 5px;">'
#         f'<p style="font-weight: bold;">{predicted_moisture_content}</p>'
#         f'</div>',
#         unsafe_allow_html=True
#     )

# # Button for issue input parameter recommendations
# if ((predicted_moisture_content[0] < input_min_LOD)|(predicted_moisture_content[0] > input_max_LOD)):
#     recc_df = reccomendation_generator(input_data_,ub_lb_df,coefficients,intercept,predicted_moisture_content,input_min_LOD,input_max_LOD)
#     if st.button("Issue Input Parameter Recommendations"):
#         st.write("Recommendation: Adjust the input parameters to bring the moisture content within the target range.")
#         st.dataframe(recc_df)

