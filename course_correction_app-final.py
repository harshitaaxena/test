#!/usr/bin/env python
# coding: utf-8

# In[63]:


##Recommendation App


# In[57]:


import pickle
from PIL import Image
import numpy as np 
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import base64

# Ignore all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

model_file = r"Meta_San_Simulator\LOD_prediction_training.pkl"
with open(model_file, 'rb') as file:  
        parameters = pickle.load(file)
        


# In[58]:


coefficients = parameters['coeffecients']
intercept= parameters['intercept']
ub_lb_df = parameters['ub_lb_df']


# In[59]:


ub_lb_df_ = pd.DataFrame({'Controllable_Parameter': ['Air Velocity','Steam Pressure','Feed_rate','Ambient Temp','Ambient Humidity','LOD_raw','Swell_vol','on_30','thru_70','thru_100'] ,
                          'lower':[29,1,1000,10,10,5,23,23,0,0 ],
                         'upper':[32,3,2000,60,60,15,35,60,7,2]})


# In[60]:


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
    max_ = Maximum allowed LOD as per user input
    
    Output :
    reccomendation_df : reccommended parameters value in each iteration'''
    rec = pd.DataFrame()
    input_data_init =input_data.copy()
    input_data_init['Iteration'] = 0
    input_data_init['LOD_achieved'] =round(predicted_moisture_content[0],2)

    coefficients_dict = dict(zip(list(input_data.columns),list(np.append(coefficients,[intercept]))))
    # List of keys to include in the subset
    controllable_params = ['Steam Inlet Temp', 'Steam Outlet Temp', 'Steam Sep Line Temp' , 'Transfer cool air temp','Transfer Air']

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
        reqd_moisture_list = list(np.arange(round(predicted_moisture_content[0],2)+(max_-min_)/2,round(min_+ (max_-min_)/2,2),(max_-min_)/2))
        input_data_init['Step Wise Target LOD Values'] = reqd_moisture_list[0]
    elif (predicted_moisture_content[0] > max_) :
        reqd_moisture_list = list(np.arange(round(min_+(max_-min_)/2,2),round(predicted_moisture_content[0]+(max_-min_)/2,2),(max_-min_)/2))[::-1]
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
        input_data_f= input_data_f.round(4)
        repeating_rows = input_data_f['LOD_achieved'] == input_data_f['LOD_achieved'].shift()
        # Delete the repeating rows from the DataFrame
        # rec = input_data_f[~repeating_rows]
        rec = input_data_f.copy()
        rec = rec[['LOD_achieved','Steam Inlet Temp','Steam Outlet Temp','Steam Sep Line Temp',
                   'Transfer cool air temp','Transfer Air','Air Velocity','Steam Pressure','Feed_rate','Ambient Temp','Ambient Humidity',
                   'LOD_raw','Swell_vol','on_30','thru_70','thru_100']]
        rec.columns = ['Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)','Steam Velocity (m/s)','Steam Pressure (bar)','Feed_rate (kg/hr)','Ambient Temp (F)','Ambient Humidity (%)',
                   'LOD Raw Husk (%)','Swell volume','% on 30','% thru70','% thru 100']
        rec=rec.round(decimals = 2)
        rec = rec.drop_duplicates()
        rec['Iteration']=np.arange(0,len(rec))
        rec =rec[['Iteration','Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)','Steam Velocity (m/s)','Steam Pressure (bar)','Feed_rate (kg/hr)','Ambient Temp (F)','Ambient Humidity (%)',
                   'LOD Raw Husk (%)','Swell volume','% on 30','% thru70','% thru 100']]
        rec = rec.reset_index().drop('index',axis=1)
        rec1 = rec[['Iteration','Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)']]
        rec1 = rec1.round(decimals = 2)
    else :
        print("No Recommendation required")
        rec1 = pd.DataFrame()
        rec = pd.DataFrame()
    return  rec1 , rec


# In[61]:


data = pd.read_csv(r'Meta_San_Simulator\sample_data.csv')


# In[62]:


data= data[[ 'Air Velocity', 'Steam Inlet Temp', 'Steam Outlet Temp',
       'Steam Sep Line Temp', 'Transfer cool air temp', 'Steam Pressure',
       'Ambient Temp', 'Ambient Humidity', 'Feed_rate', 'LOD_raw', 'Swell_vol',
       'on_30', 'thru_70', 'thru_100','Transfer Air', 'Moisture Content']]


# In[64]:


def sensitivity_graph(data, input_data_, coefficients, intercept, ub_lb_df, p):
    coefficients_dict = dict(zip(list(input_data_.columns), list(np.append(coefficients, [intercept]))))
    # Convert values to lists
    coefficients_dict = {key: [value] for key, value in coefficients_dict.items()}
    coefficients_df = pd.DataFrame(coefficients_dict)
    a = np.dot(np.array(input_data_.drop(p, axis=1))[0], np.array(coefficients_df.drop(p, axis=1))[0])
    slope = float(coefficients_df[p])
    intercept = float(a)

    # Define the equation
    def equation(x, slope, intercept):
        return intercept + slope * x

    x = np.arange(int(ub_lb_df[ub_lb_df['Controllable_Parameter'] == p].lower),
                  int(ub_lb_df[ub_lb_df['Controllable_Parameter'] == p].upper), 0.1)
    y = equation(x, slope, intercept)

    # Dynamic user input for x
    user_x = float(input_data_[p])
    # Calculate y for the user input x
    user_y = equation(user_x, slope, intercept)

    # Plot the equation
    # Plot the user input point
    fig, ax = plt.subplots()
    ax.plot(user_x, user_y, 'ro')  # 'ro' for red circles
    ax.annotate(f'({round(user_x, 2)}, {round(user_y, 2)})', (user_x, user_y), xytext=(-25, 15),
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    ax.plot(x, y)

    if p =='Air Velocity':
        ax.set_xlabel('Steam Velocity (m/s)')
    elif p=='Steam Inlet Temp':
        ax.set_xlabel('Steam Inlet Temp (C)')
    elif p== 'Steam Outlet Temp':
         ax.set_xlabel('Steam Outlet Temp (C)')
    elif p== 'Steam Sep Line Temp':
         ax.set_xlabel('Steam Sep Line Temp (C)')
    elif p== 'Transfer cool air temp':
        ax.set_xlabel('Cool Air (C)')
    elif p== 'Transfer Air':
        ax.set_xlabel('Transfer Air (C)')
    elif p== 'Steam Pressure':
        ax.set_xlabel('Steam Pressure (bar)')
    elif p== 'Ambient Temp':
        ax.set_xlabel('Ambient Temp (F)')
    elif p== 'Ambient Humidity':
        ax.set_xlabel('Ambient Humidity (%)')
    elif p== 'Feed_rate':
        ax.set_xlabel('Feed Rate (kg/hr)')
    elif p== 'LOD_raw':
        ax.set_xlabel('Raw Husk LOD %')
    elif p== 'Swell_vol':
        ax.set_xlabel('Swell Volume')
    elif p== 'on_30':
        ax.set_xlabel('% on 30 mesh')
    elif p== 'thru_70':
        ax.set_xlabel('% thru 70 mesh')
    else:
        ax.set_xlabel('% thru 100 mesh')
        
    ax.set_ylabel('LOD %')
    ax.grid(True)
    st.pyplot(fig)
    
    #return fig


# In[65]:


def alerts(number,target_min,target_max):
    if number < target_min or number > target_max:
#         st.markdown('<p style="font-weight:bold; color:red; animation: blink 1s infinite;">'
#                     f'<i class="fas fa-exclamation-triangle"></i> {number}</p>', unsafe_allow_html=True)
        st.error('Outside Specified Range : '+str(target_min)+'-'+str(target_max))


# In[66]:


def format_excel(df): 
    change_description = [''] * len(df)  # Blank entries for all rows
    previous_values = {}

    for index, row in df.iterrows():
        variable_changes = []

        for column in ['Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)']:
            if column not in previous_values:
                previous_values[column] = row[column]
            elif row[column] > previous_values[column]:
                change = round(row[column] - previous_values[column], 2)
                variable_changes.append(f"Increase {column} by {change} Degree C")
                previous_values[column] = row[column]
            elif row[column] < previous_values[column]:
                change = round(previous_values[column] - row[column], 2)
                variable_changes.append(f"Decrease {column} by {change} Degree C")
                previous_values[column] = row[column]

        change_description[index] = ', '.join(variable_changes)

    df['Change Description'] = change_description
    return df


# In[67]:


def highlight_cells(row, selected_columns):
    styles = ['' for _ in row]  # Initialize an empty style list for each cell in the row
    for column in selected_columns:
        style = ['background-color: green' if val != prev_val and col == column else ''
                 for val, prev_val, col in zip(row, row.shift(), row.index)]
        styles = [s or st for s, st in zip(styles, style)]  # Apply the style to the corresponding cells
    return styles


# In[68]:


# Define page names
PAGE_1 =  "Sanitization"
PAGE_2 =  "Milling"
PAGE_3 =  "Agglomeration Sugar"
PAGE_4 =  "Agglomeration Sugar Free"

# Create navigation buttons
st.set_page_config(page_title="Simulator : Course Correction Sanitization Application",layout="wide")
nav_option = st.sidebar.radio("Simulator : Course Correction", (PAGE_1, PAGE_2 ,PAGE_3,PAGE_4))

# Display content based on selected page
if nav_option == PAGE_1:
    st.title("Course Correction : Sanitization")
    # Define the options for the radio button
    options = ('LOD Prediction','Recommendation','Sensitivity Analysis')

    # Initialize the selected option
    selected_option = st.session_state.page_selection if 'page_selection' in st.session_state else options[0]

    # Create the radio button with a horizontal layout
    selected_option = st.radio("", options, index=options.index(selected_option),horizontal=True, 
                               key="navigation", format_func=lambda x: x, help="Page Navigation")

    # Store the selected option in session state
    st.session_state.page_selection = selected_option

    # Show the corresponding page based on the selection
    if selected_option == 'Sensitivity Analysis':
        st.title("Sensitivity Analysis")
        
        st.subheader("Controllable Parameters")
        # First row with four columns
        col1_row1, col2_row1, col3_row1, col4_row1,col5_row1 = st.columns(5)
        with col1_row1:
            input_steam_inlet_temp = st.slider('Steam In Temp(C)' ,min_value = 147 , max_value = 153)
        with col2_row1:
            input_steam_outlet_temp = st.slider('Steam Out Temp(C)' ,min_value = 136 , max_value = 145)
        with col3_row1:
            input_steam_sep_line_temp = st.slider( 'Steam Sep(C)' ,min_value = 115 , max_value = 125)    
        with col4_row1:
            input_transfer_cool_air_temp = st.slider('Cooling Air(C)' ,min_value = 63 , max_value = 125)
        with col5_row1:
            input_transfer_air = st.slider('Transfer Air(C)' ,min_value = 40 , max_value = 125)
            
        st.subheader("External Parameters")
        col1_row3, col2_row3  = st.columns(2)
        with col1_row3:
            ambient_temp= st.slider('Ambient Temp(F)' ,min_value = 10 , max_value = 60)  
        with col2_row3:
            ambient_humidity = st.slider('Ambient Humidity(%) ' ,min_value = 10 , max_value = 60)
            
        st.subheader("Raw Material Properties")
        col1_row4, col2_row4 , col3_row4 , col4_row4 ,col5_row4 = st.columns(5)
        with col1_row4:
            LOD_raw = st.slider('Raw Husk LOD' ,min_value = 5 , max_value = 15)    
        with col2_row4:
            Swell_vol = st.slider('Swell Volume' ,min_value = 23 , max_value = 35)
            # Add your content for Column 2
        with col3_row4:
            on_30 = st.slider( '% on 30 mesh' ,min_value = 23 , max_value = 60)        
        with col4_row4:
            thru_70 = st.slider( '% thru 70' ,min_value = 0 , max_value = 7)     
        with col5_row4:
            thru_100 = st.slider( '% thru 100' ,min_value = 0 , max_value = 2)
            
        st.subheader("Other Parameters")
        col1_row2, col2_row2 , col3_row2 = st.columns(3)
        with col1_row2:
            feed_rate = st.slider( 'Feed Rate (kg/hr)' ,min_value = 1000 , max_value = 2000)
        with col2_row2:
            air_velocity = st.slider('Steam Velocity (m/s)' ,min_value = 29 , max_value = 32)   
        with col3_row2:
            steam_pressure = st.slider('Steam Pressure (bar)' ,min_value = 2.6 , max_value = 2.8)
   
        input_data_ = pd.DataFrame({
        'Air Velocity': [float(air_velocity)],
        'Steam Inlet Temp': [float(input_steam_inlet_temp)],
        'Steam Outlet Temp': [float(input_steam_outlet_temp)],
        'Steam Sep Line Temp': [float(input_steam_sep_line_temp)],
        'Transfer cool air temp':[float(input_transfer_cool_air_temp)],
        'Steam Pressure': [float(steam_pressure)],
        'Ambient Temp': [float(ambient_temp)],
        'Ambient Humidity':[float(ambient_humidity)],
        'Feed_rate':[float(feed_rate)],
        'LOD_raw':[float(LOD_raw)],
        'Swell_vol':[float(Swell_vol)],
        'on_30':[float(on_30)],
        'thru_70':[float(thru_70)],
        'thru_100':[float(thru_100)],
        'Transfer Air':[float(input_transfer_air)]})
    
        input_data_['const']=1
        #predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
        # Display the predictions
        st.sidebar.subheader("LOD Sanitized Husk")
        predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
        # Display the predictions
        styled_text = f"<span style='font-weight:bold;color:green'>{round(predicted_moisture_content[0],6):.2f}%</span>"
        st.sidebar.markdown(styled_text, unsafe_allow_html=True)
        with col1_row1:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df,'Steam Inlet Temp')
        with col2_row1:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df,'Steam Outlet Temp')
        with col3_row1:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df,'Steam Sep Line Temp')
        with col4_row1:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df,'Transfer cool air temp')
        with col5_row1:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df,'Transfer Air')
        with col1_row2:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Feed_rate')
        with col2_row2:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Air Velocity')
        with col3_row2:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Steam Pressure')
        with col1_row3:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Ambient Temp')
        with col2_row3:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Ambient Humidity')
        with col1_row4:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'LOD_raw')
        with col2_row4:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'Swell_vol')
        with col3_row4:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'on_30')
        with col4_row4:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'thru_70')
        with col5_row4:
            sensitivity_graph(data,input_data_,coefficients,intercept,ub_lb_df_,'thru_100')
            
    elif selected_option == 'LOD Prediction':
        # Create a Streamlit web app
        st.title("LOD Prediction")
        # Upload manufacturing image
        image = Image.open(r"Meta_San_Simulator\manufacturing_img.jpeg")
        st.image(image, use_column_width=True)
        st.subheader("Controllable Parameters")
        # User input for the controllable independent variables
        col1 , col2 , col3 , col4 , col5 = st.columns(5) 
        with col1:
            input_steam_inlet_temp = st.number_input("Steam In Temp (C): " , value = 153)
            alerts(float(input_steam_inlet_temp),147,153)
        with col2:
            input_steam_outlet_temp = st.number_input("Steam Out Temp (C): ",value = 145)
            alerts(float(input_steam_outlet_temp),136,145)
        with col3:
            input_steam_sep_line_temp = st.number_input("Steam Sep (C): ",value = 145)
            alerts(float(input_steam_sep_line_temp),115,125)
        with col4:
            input_transfer_cool_air_temp = st.number_input("Cooling Air (C):",value = 90)
            alerts(float(input_transfer_cool_air_temp),63,125)
        with col5:
            input_transfer_air_temp = st.number_input("Transfer Air (C):",value = 63)
            alerts(float(input_transfer_cool_air_temp),40,125)
        # Display subheading for external temperature
        st.subheader("External Parameters")
        col1 , col2 = st.columns(2)
        with col1:
            input_ambient_temp = st.number_input("Ambient Temperature (F):" , value = 30)
        with col2:
            ambient_humidity = st.number_input("Ambient Humidity (%):" , value = 30)

        st.subheader('Raw Material Properties')
         #Display subheading for raw material characteristics
        col1, col2 ,col3,col4,col5 = st.columns(5)
        # Display the raw material charecteristics
        with col1:
            LOD_raw_husk = st.number_input("LOD Raw Husk: " , value = 9.6)
        with col2:
            swell = st.number_input("Swell Volume " , value = 29.87)
        with col3:
            on_30 = st.number_input("% on 30 mesh: " , value = 42.2)
        with col4:
            thru_70 = st.number_input("% thru 70 mesh: " , value = 0.10)
        with col5:
            thru_100 = st.number_input(" % thru 100 mesh: " , value = 0.5)
        st.subheader('Other Parameters')
         #Display subheading for fixed input parameters
        col1 , col2, col3 = st.columns(3)
        with col1:
            avg_feed_rate = st.number_input("Feed Rate (kg/hr): " ,min_value= 1000,max_value=2000 , value = 1000)
            alerts(float(avg_feed_rate),1000,2000)
        with col2:
            avg_steam_pressure = st.number_input("Steam Pressure (bar): " , min_value=2.6 ,max_value=2.8 ,value = 2.6)
            alerts(float(avg_steam_pressure),2.6,2.8)
        with col3:
            avg_air_velocity = st.number_input("Steam Velocity (m/s): " ,min_value=29 ,max_value=32 , value = 29)
            alerts(float(avg_air_velocity),29,32)
        # Create a dataframe with the user input and fixed values
        input_data_ = pd.DataFrame({
            'Air Velocity': [float(avg_air_velocity)],
            'Steam Inlet Temp': [float(input_steam_inlet_temp)],
            'Steam Outlet Temp': [float(input_steam_outlet_temp)],
            'Steam Sep Line Temp': [float(input_steam_sep_line_temp)],
            'Transfer cool air temp':[float(input_transfer_cool_air_temp)],
            'Steam Pressure': [float(avg_steam_pressure)],
            'Ambient Temp': [float(input_ambient_temp)],
            'Ambient Humidity':[float(ambient_humidity)],
            'Feed_rate':[float(avg_feed_rate)],
            'LOD_raw':[float(LOD_raw_husk)],
            'Swell_vol':[float(swell)],
            'on_30':[float(on_30)],
            'thru_70':[float(thru_70)],
            'thru_100':[float(thru_100)],
            'Transfer Air':[float(input_transfer_air_temp)]
        })
        input_data_['const'] = 1.0 #sm.add_constant(input_data)
        # Predict the moisture content using the trained model
        # Display the predicted moisture content
        if st.button('Run'):
            st.subheader("LOD Sanitized Husk")
            predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
            # Display the predictions
            styled_text = f"<span style='font-weight:bold;color:green'>{round(predicted_moisture_content[0],2):.2f}%</span>"
            st.markdown(styled_text, unsafe_allow_html=True)
        #st.write(round(predicted_moisture_content[0],5))
            #st.write('You can optimize the output if it is not in desired range')
            #alert_specif_range_max(input_max_LOD,7.6)
        #st.session_state.number1 = predicted_moisture_content[0]
        predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
        st.session_state['stored_np_array'] = predicted_moisture_content
        st.session_state.dataframe = input_data_
    else:            
        st.title('Recommendation')
        input_data_=st.session_state.dataframe
        input_data_1=input_data_.iloc[0]
        st.subheader('Parameter Summary')
        st.write("You can edit the Parameters on the LOD Prediction Screen")
        st.subheader("Controllable Parameters")
        # User input for the controllable independent variables
        col1 , col2 , col3 , col4 , col5 = st.columns(5) 
        with col1:
            st.write("Steam In Temp (C): " ,input_data_1['Steam Inlet Temp'])
        with col2:
            st.write("Steam Out Temp (C): ",input_data_1['Steam Outlet Temp'])
        with col3:
            st.write("Steam Sep (C): ",input_data_1['Steam Sep Line Temp'])
        with col4:
            st.write("Cooling Air (C):",input_data_1['Transfer cool air temp'])
        with col5:
            st.write("Transfer Air (C):",input_data_1['Transfer Air'])
        # Display subheading for external temperature
        st.subheader("External Parameters")
        col1 , col2 = st.columns(2)
        with col1:
            st.write("Ambient Temperature (F):" , input_data_1['Ambient Temp'])
        with col2:
            st.write("Ambient Humidity (%):" , input_data_1['Ambient Humidity'])

        st.subheader('Raw Material Properties')
         #Display subheading for raw material characteristics
        col1, col2 ,col3,col4,col5 = st.columns(5)
        # Display the raw material charecteristics
        with col1:
            st.write("LOD Raw Husk: " , input_data_1['LOD_raw'])
        with col2:
            st.write("Swell Volume " , input_data_1['Swell_vol'])
        with col3:
            st.write("% on 30 mesh: " , input_data_1['on_30'])
        with col4:
            st.write("% thru 70 mesh: " , input_data_1['thru_70'])
        with col5:
            st.write(" % thru 100 mesh: " , input_data_1['thru_100'])
        st.subheader('Other Parameters')
         #Display subheading for fixed input parameters
        col1 , col2, col3 = st.columns(3)
        with col1:
            st.write("Feed Rate (kg/hr): " ,input_data_1['Feed_rate'])
        with col2:
             st.write("Steam Pressure (bar): " , input_data_1['Steam Pressure'])
        with col3:
            st.write("Steam Velocity (m/s): " ,input_data_1['Air Velocity'])
        
        col1 , col2 ,col3 =st.columns([3,1,1])
        with col1:
            st.subheader('Predicted LOD Sanitized Husk')
            predicted_moisture_content= st.session_state.get('stored_np_array', np.array([]))
            #predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
             # Display the predictions
            styled_text = f"<span style='font-weight:bold;color:green'>{round(predicted_moisture_content[0],2):.2f}%</span>"
            st.markdown(styled_text, unsafe_allow_html=True)
        with col2:
            input_min_LOD = float(st.number_input("Minimum LOD % :",value = 5))
            #alerts(input_min_LOD,3.9,7.6)
        with col3:
            input_max_LOD = float(st.number_input("Maximum LOD %:",value = 8))
            #alerts(input_max_LOD,3.9,7.6)
        st.write('You can Optimize the output if it is not in the desired range')
        if st.button('Run and Optimize'):
        # Button for issue input parameter recommendations
            if ((predicted_moisture_content[0] < input_min_LOD)|(predicted_moisture_content[0] > input_max_LOD)):
                recc_df , recc_all = reccomendation_generator(input_data_,ub_lb_df,coefficients,intercept,predicted_moisture_content,input_min_LOD,input_max_LOD)
                recc_df = format_excel(recc_df)
                selected_columns= ['Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)']
                
                recc_df = recc_df.round(decimals=2)
                # Apply the style to the DataFrame for each selected column
                
                #styled_df = recc_df.style.apply(highlight_cells, selected_columns=selected_columns, axis=1)
                recc_all_ = format_excel(recc_all)
                recc_all_ = recc_all_.round(decimals =2)
              
                st.write("Recommendation: Adjust the controllable input parameters to bring LOD within the target range given bounds on parameters")
                st.dataframe(recc_df.set_index(recc_df.columns[0]))
                st.write("Maximum Iterations Reached")
                if ((recc_all_['Updated LOD % '].iloc[-1]>input_max_LOD)|(recc_all_['Updated LOD % '].iloc[-1]<input_min_LOD)):
                    st.error('Convergence to range cannot be attained given Target Ranges for Parameters')
                else:
                    st.success('Convergence to Target Range attained')
                # Add a download button
                csv = recc_all_.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write('No reccomendation required')                    
elif nav_option == PAGE_2:
    st.title("Course Correction : Milling")
    st.write("This page is a place holder for milling")
elif nav_option == PAGE_3:
    st.title("Course Correction : Agglomeration Sugar")
    st.write("This page is a place holder for Agglomeration Sugar")
else :
    st.title("Course Correction : Agglomeration Sugar Free")
    st.write("This page is a place holder for Agglomeration Sugar Free")



# In[ ]:




