from library import *

# heuristic logic
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
    #print(sorted_coefficients_dict)

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
            #print(reccommendation_df)
        input_data_f = pd.concat([input_data_init,input_data1], ignore_index= True)
        print("Maximum Convergence Limit Reached")
        input_data_f= input_data_f.round(4)
        repeating_rows = input_data_f['LOD_achieved'] == input_data_f['LOD_achieved'].shift()
        # Delete the repeating rows from the DataFrame
        # rec = input_data_f[~repeating_rows]
        rec = input_data_f.copy()
        rec = rec[['Iteration','LOD_achieved','Steam Inlet Temp','Steam Outlet Temp','Steam Sep Line Temp',
                   'Transfer cool air temp','Transfer Air','Air Velocity','Steam Pressure','Feed_rate','Ambient Temp','Ambient Humidity',
                   'LOD_raw','Swell_vol','on_30','thru_70','thru_100']]
        rec.columns = ['Iteration','Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)','Steam Velocity (m/s)','Steam Pressure (bar)','Feed_rate (kg/hr)','Ambient Temp (F)','Ambient Humidity (%)',
                   'Raw Husk LOD (%)','Swell volume(ml)','% on 30','% thru70','% thru 100']
        rec=rec.round(decimals = 2)
        rec = rec.drop_duplicates()
        rec['Iteration']= (rec['Updated LOD % '].diff() != 0).cumsum() - 1
        #rec['Iteration']=np.arange(0,len(rec))
        rec =rec[['Iteration','Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)','Steam Velocity (m/s)','Steam Pressure (bar)','Feed_rate (kg/hr)','Ambient Temp (F)','Ambient Humidity (%)',
                   'Raw Husk LOD (%)','Swell volume(ml)','% on 30','% thru70','% thru 100']]
        rec = rec.reset_index().drop('index',axis=1)
        rec1 = rec[['Iteration','Updated LOD % ','Steam In Temp (C)','Steam Out Temp (C)','Steam Sep (C)',
                   'Cooling Air (C)','Transfer Air (C)']]
        rec1 = rec1.round(decimals = 2)
        r1 = pd.DataFrame()
        for x in list(rec['Iteration'].unique()):
            recc_df_x=rec[rec.Iteration==x]
            r = recc_df_x.iloc[-1].to_frame().transpose()
            r1 =pd.concat([r1,r],ignore_index=True)
    else :
        print("No Recommendation required")
        rec1 = pd.DataFrame()
        rec = pd.DataFrame()
    return  rec1 , r1