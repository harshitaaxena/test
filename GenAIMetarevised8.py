#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
import pandas as pd
import streamlit as st
# Create navigation buttons
st.set_page_config(page_title="Operator Training Application",layout="wide")
from streamlit_chat import message
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import base64
import re
import time
from io import BytesIO
from typing import Any, Dict, List
from gtts import gTTS
import openai
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent ,create_pandas_dataframe_agent
from langchain.chains import RetrievalQA , ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory , ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from config import *

# In[35]:


# Ignore all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# Define a function for the embeddings
@st.cache_data
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("Document Processing"):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Processing Completed", icon="✅")
    return index

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
        ax.set_xlabel('Swell Volume (ml)')
    elif p== 'on_30':
        ax.set_xlabel('% on 30 mesh')
    elif p== 'thru_70':
        ax.set_xlabel('% thru 70 mesh')
    else:
        ax.set_xlabel('% thru 100 mesh')
        
    ax.set_ylabel('LOD %')
    ax.grid(True)
    st.pyplot(fig)
    


def alerts(number,target_min,target_max):
    if number < target_min or number > target_max:
#         st.markdown('<p style="font-weight:bold; color:red; animation: blink 1s infinite;">'
#                     f'<i class="fas fa-exclamation-triangle"></i> {number}</p>', unsafe_allow_html=True)
        st.error('Outside Specified Range : '+str(target_min)+'-'+str(target_max))

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

    df['Recommendation'] = change_description
    return df



# In[39]:


def number(text):
    # Extract the number from the text
    number = re.search(r'\d+', text)   
    if number:
        # Convert the extracted number to an integer
        num = int(number.group())
        return num
    else:
        return None
    
def sim(text,df):
    if 'increase' in text:
        if 'Steam Inlet Temp' in text:
            num = number(text)
            df['Steam Inlet Temp'] = df['Steam Inlet Temp'] + num
        if 'Steam Outlet Temp' in text:
            num = number(text)
            df['Steam Outlet Temp'] = df['Steam Outlet Temp'] + num
        if 'Steam Sep Line Temp' in text:
            num = number(text)
            df['Steam Sep Line Temp'] = df['Steam Sep Line Temp'] + num
        if 'Transfer cool air temp' in text:
            num = num(text)
            df['Transfer cool air temp'] = df['Transfer cool air temp'] + num
        if 'Transfer cool air temp' in text:
            num = num(text)
            df['Transfer Air'] = df['Transfer Air'] + num
    if 'decrease' in text:
        if 'Steam Inlet Temp' in text:
            num = number(text)
            df['Steam Inlet Temp'] = df['Steam Inlet Temp'] - num
        if 'Steam Outlet Temp' in text:
            num = number(text)
            df['Steam Outlet Temp'] = df['Steam Outlet Temp'] - num
        if 'Steam Sep Line Temp' in text:
            num = number(text)
            df['Steam Sep Line Temp'] = df['Steam Sep Line Temp'] - num
        if 'Transfer cool air temp' in text:
            num = number(text)
            df['Transfer cool air temp'] = df['Transfer cool air temp'] - num
        if 'Transfer cool air temp' in text:
            num = number(text)
            df['Transfer Air'] = df['Transfer Air'] - num
    if 'change' in text:
        if 'Steam Inlet Temp' in text:
            num = number(text)
            df['Steam Inlet Temp'] = num
        if 'Steam Outlet Temp' in text:
            num = number(text)
            df['Steam Outlet Temp'] = num
        if 'Steam Sep Line Temp' in text:
            num = number(text)
            df['Steam Sep Line Temp'] =  num
        if 'Transfer cool air temp' in text:
            num = number(text)
            df['Transfer cool air temp'] =  num
        if 'Transfer cool air temp' in text:
            num = number(text)
            df['Transfer Air'] = num
    return df


# In[40]:


model_file = r"LOD_prediction_training.pkl" #read model parameters
with open(model_file, 'rb') as file:  
        parameters = pickle.load(file)
coefficients = parameters['coeffecients']
intercept= parameters['intercept']
ub_lb_df = parameters['ub_lb_df']
ub_lb_df_ = pd.DataFrame({'Controllable_Parameter': ['Air Velocity','Steam Pressure','Feed_rate','Ambient Temp','Ambient Humidity','LOD_raw','Swell_vol','on_30','thru_70','thru_100'] ,
                          'lower':[29,1,1000,10,10,5,23,23,0,0 ],
                         'upper':[32,3,2000,60,60,15,35,60,7,2]})

data = pd.read_csv(r'sample_data.csv') #read sample data
data= data[['Air Velocity',
       'Steam Inlet Temp', 'Steam Outlet Temp', 'Steam Sep Line Temp',
       'Transfer cool air temp', 'Steam Pressure', 'Ambient Temp',
       'Ambient Humidity', 'Feed_rate', 'LOD_raw', 'Swell_vol', 'on_30',
       'thru_70', 'thru_100', 'Transfer Air', 'Moisture Content']]  


# In[41]:


df=data.copy()
f = pd.read_csv(r'plot_data.csv')

# In[44]:


#streamlit app 
# Define page names
PAGE_1 =  "Home"
PAGE_2 =  "Update SOP Document"
PAGE_3 =  "Simulator"
PAGE_4 =  "Chat"


nav_option = st.sidebar.radio("Operator Training Application", (PAGE_1, PAGE_2 ,PAGE_3,PAGE_4))

# Display content based on selected page
if nav_option == PAGE_1:
    st.title("Home")
    # Create the figure and axes
    # Define the placard content
    placards = [
        {"title": "Production Volume", "content": "200k Tonnes"},
        {"title": "Production Line Efficiency", "content": "40%"},
        {"title": "Capacity Utilization", "content": "120%"},
        {"title": "Loss on Drying", "content": "15%"},
        {"title": "Total Scrap Rate", "content": "30%"},
    ]

    # Set the layout to display columns
    col1, col2, col3, col4, col5 = st.columns(5)


    # Display content in each column
    with col1:
        st.success(placards[0]["title"])
        st.success(placards[0]["content"])

    with col2:
        st.success(placards[1]["title"])
        st.success(placards[1]["content"])

    with col3:
        st.success(placards[2]["title"])
        st.success(placards[2]["content"])

    with col4:
        st.success(placards[3]["title"])
        st.success(placards[3]["content"])

    with col5:
        st.success(placards[4]["title"])
        st.success(placards[4]["content"])

# =============================================================================
#     fig, ax1 = plt.subplots(figsize=(10, 4))
#     ax2 = ax1.twinx()
#     
#     # Plot the bar graph for LOD%
#     ax1.bar(f['Day'], f['LOD%'], color='blue', alpha=0.5)
#     ax1.set_ylabel('LOD%')
#     ax1.set_xlabel('Day')
#     
#     # Plot the line chart for Efficiency%
#     ax2.plot(f['Day'],f['Efficiency%'], color='red', marker='o')
#     ax2.set_ylabel('Efficiency%')
#     
#     # Set the title
#     plt.title('LOD% and Efficiency%')
#     
#     # Adjust the layout
#     plt.tight_layout()
# =============================================================================
    
    
    days = f['Day']
    lod = f['LOD%']
    efficiency = f['Efficiency%']
    efficiency = [int(d.strip('%')) for d in efficiency]
    
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    ax1.bar(days, lod, color='blue',alpha=0.6)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('LOD %')
    
    ax2 = ax1.twinx()
    ax2.plot(days, efficiency, color='green', marker='o')
    ax2.set_ylabel('Efficiency %')
    
    ax1.legend(['LOD'],loc='upper left')
    ax2.legend(['Efficiency'],loc='upper right')
    
    plt.title('LOD% and Efficiency')
    plt.grid(True)
    plt.show()
    
    # Display the chart in Streamlit
    st.pyplot(fig)
    st.write('Loss on Drying refers to Moisture content %')
    
elif nav_option == PAGE_2:
        st.title("Update SOP Document")
        # Allow the user to upload a PDF file
        uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])
        st.session_state.uploaded_file = uploaded_file
        
elif nav_option == PAGE_3 :
    st.title("Simulator")
    # Define the options for the radio button
    options = ('Prediction','Recommendation','Sensitivity Analysis')

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
            LOD_raw = st.slider('Raw Husk LOD %' ,min_value = 5 , max_value = 15)    
        with col2_row4:
            Swell_vol = st.slider('Swell Volume (ml)' ,min_value = 23 , max_value = 35)
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
        st.sidebar.subheader("Output %")
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
    elif selected_option == 'Prediction':
        # Create a Streamlit web app
        st.title("Simulating Results for a New Line")
       
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
            input_ambient_temp = st.number_input("Ambient Temperature (F):" , value = 90)
        with col2:
            ambient_humidity = st.number_input("Ambient Humidity (%):" , value = 30)            
        st.subheader('Raw Material Properties')
         #Display subheading for raw material characteristics
        col1, col2 ,col3,col4,col5 = st.columns(5)
        # Display the raw material charecteristics
        with col1:
            LOD_raw_husk = st.number_input("Raw Husk LOD%: " , value = 9.6)
        with col2:
            swell = st.number_input("Swell Volume (ml) " , value = 29.87)
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
        input_data_1=input_data_.iloc[0]
        if st.button('Run'):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
    
            with st.spinner("Loading Output For the New Production Line.."):
                for percent_complete in range(100):
                    time.sleep(0.04)  # Simulate loading time
                    progress_bar.progress(percent_complete + 1)
                    status_text.text(f"Loading... {percent_complete + 1}%")
    
            # Display loading completed message
            st.success("Loading completed!")

            st.subheader("Predicted Output for New Line")
            
            # Upload manufacturing image
            image = Image.open(r"manufacturing_img.jpeg") #read img
            # Create a draw object
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf",size= 12)
            # Specify the coordinates to place the number

            # Write the number on the image
            
            text = "Steam In Temp (C): " + str(input_data_1['Steam Inlet Temp'])
            draw.text((0,25), text, fill="red", font=font)
            
            text = "Steam Out Temp (C): " + str(input_data_1['Steam Outlet Temp'])
            draw.text((0, 40), text, fill="red", font=font)
            
            text = "Steam Sep Line Temp (C): "  + str(input_data_1['Steam Sep Line Temp'])
            draw.text((0,55), text, fill="red", font=font)
            
            text = "Transfer cool air temp(C):" + str(input_data_1[ 'Transfer cool air temp'])
            draw.text((0,70), text, fill="red", font=font)
            
            text = "Transfer Air (C):"  + str(input_data_1[ 'Transfer Air'])
            draw.text((0,85), text, fill="red", font=font)
            
            # Get the image's width and height
            width, height = image.size
          
            x = 0 
            y = 5
            
            predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
            font = ImageFont.truetype("arial.ttf",size= 17)
            text = "Predicted LOD Content :" + str(round( predicted_moisture_content[0],2)) +"%"
            draw.text((x,y), text, fill="green",font=font)
            st.image(image, use_column_width=True)
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
        st.session_state.data = input_data_
        input_data_1=input_data_.iloc[0]
        st.subheader('Parameter Summary')
        st.write("You can edit the Parameters on the Prediction Screen")
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
            st.write("Raw Husk LOD%: " , input_data_1['LOD_raw'])
        with col2:
            st.write("Swell Volume (m/l): " , input_data_1['Swell_vol'])
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
            st.subheader('Predicted Output ')
            predicted_moisture_content= st.session_state.get('stored_np_array', np.array([]))
            #predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))
             # Display the predictions
            styled_text = f"<span style='font-weight:bold;color:green'>{round(predicted_moisture_content[0],2):.2f}%</span>"
            st.markdown(styled_text, unsafe_allow_html=True)
        with col2:
            input_min_LOD = float(st.number_input("Minimum  % :",value = 5))
            #alerts(input_min_LOD,3.9,7.6)
        with col3:
            input_max_LOD = float(st.number_input("Maximum %:",value = 8))
            #alerts(input_max_LOD,3.9,7.6)
        st.write('You can Optimize the output if it is not in the desired range')
        
        #
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
                recc_all_['Iteration']=recc_all_['Iteration'].astype(int)
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
                    
else:
    st.title("Chat")
    st.markdown(
    """ 
        ####  🗨️ Chat with your Training Document 📜  
        
        """
        )
  
    #st.write(predicted_moisture_content)
    sound_file = BytesIO()
    text ="Hello I am your AI powered Training Assistant.You can ask queries relevant to the SOP document as well as the simulation tool !!"
    tts = gTTS(text, lang='en' ,slow = 'False')
    tts.write_to_fp(sound_file)
    st.sidebar.audio(sound_file)
    uploaded_file = st.session_state.uploaded_file
    if uploaded_file:
            name_of_file = uploaded_file.name
            doc = parse_pdf(uploaded_file)
            pages = text_to_docs(doc)
            if pages:
                # Allow the user to select a page and view its content
                with st.expander("Show Page Content", expanded=False):
                    page_sel = st.number_input(
                        label="Select Page", min_value=1, max_value=len(pages), step=1
                    )
                    pages[page_sel - 1]
                api = api
                if api:
                    # Test the embeddings and save the index in a vector database
                    index = test_embed()
            #st.session_state.index = index
                    with st.sidebar.expander("FAQs",expanded = False):
                        st.write("Q : What is the scope ?")
                        st.write("Q : Give on Overview of the document")
                        st.write("Q : Explain the entire process that Raw Husk goes through ")
                        st.write("Q : Where is the Steam Inlet Temperature measured ?")
                        st.write("Q : Tell me about the Steam Cyclone ")
                        st.write("Q : Explain what happens in the Steam Sep cyclone and Drying Cyclone ")
                        st.write("Q : What is the milling process ?")
                        st.write("Q : What is the output parameter ?")
                        st.write("Q : What are the input parameters ?")
                        st.write("Q : What are the controllable parameters ?")
                        st.write("Q : What are the Raw Material Properties ?")
                        st.write("Q : What is LOD?")
                        st.write("Q : What are the ranges for controllable parameters?")
                    
                        
                    qa = RetrievalQA.from_chain_type(
                        llm=OpenAI(openai_api_key=api),
                        chain_type = "map_reduce",
                        retriever=index.as_retriever(),
                    )
                
                    # Set up the conversational agent
                    tools = [
                        Tool(
                            name="State of Union QA System",
                            func=qa.run,
                            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                        )
                    ]
                    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                                You have access to a single tool:"""
                    suffix = """Begin!"
                
                    {chat_history}
                    Question: {input}
                    {agent_scratchpad}"""
                
                    prompt = ZeroShotAgent.create_prompt(
                        tools,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input", "chat_history", "agent_scratchpad"],
                    )
                    
                    if "memory" not in st.session_state:
                                st.session_state.memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history") 
                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = ['Hello Ask me anything about the manufacturing process']
                    if 'past' not in st.session_state:
                        st.session_state['past'] = ['Hey!'] 
                
                    #container for the chat history
                    response_container = st.container()
                    #container for the user's text input
                    container = st.container()
                
                    llm_chain = LLMChain(
                        llm=OpenAI(
                            temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                        ),
                        prompt=prompt,
                    )
                    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                    agent_chain = AgentExecutor.from_agent_and_tools(
                        agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                    )
                
                    with container:
                        # Allow the user to enter a query and generate a response
                        query = st.text_input(
                            "**What's on your mind?**",
                            placeholder="Hello I am your Training Assistant! . Ask me anything from SOP document or simulation process {}".format(name_of_file), key='input') 
                        if query :
                            with st.spinner(
                                "Generating Answer to your Query : `{}` ".format(query)
                            ):
                                res = agent_chain.run(query)
                                #st.info(res, icon="🤖")
                                sound_file = BytesIO()
                                tts = gTTS(res, lang='en')
                                tts.write_to_fp(sound_file)
                                st.audio(sound_file)
                                st.session_state['past'].append(query)
                                st.session_state['generated'].append(res)
                
                                if st.session_state['generated']:
                                    with response_container:
                                        for i in range(len(st.session_state['generated'])):
                                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                                            message(st.session_state["generated"][i], key=str(i))
                
                
                                    # Allow the user to view the conversation history and other information stored in the agent's memory
                                    #with st.expander("History/Memory"):
                                        #st.session_state.memory
                                        #message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
    
    
# In[ ]:


# =============================================================================
#                            if 'increase' in query or 'decrease' in query or 'change' in query:
#                                  with st.spinner(
#                                  "Generating Answer to your Query : `{}` ".format(query)
#                                   ):
#                                      input_data_= sim(query,input_data_)
#                                      prediction = round(np.dot(input_data_, np.append(coefficients,[intercept]))[0],2)
#                                      #st.info(res, icon="🤖")
#                                      res = "Predicted Output will be " + prediction.astype(str) +" % "
#                                      sound_file = BytesIO()
#                                      tts = gTTS(res, lang='en')
#                                      tts.write_to_fp(sound_file)
#                                      st.audio(sound_file)
#                                      st.session_state['past'].append(query)
#                                      st.session_state['generated'].append(res) 
#                                      if st.session_state['generated']:
#                                          with response_container:
#                                              for i in range(len(st.session_state['generated'])):
#                                                  message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                                                  message(st.session_state["generated"][i], key=str(i))
#                      
# #                                         # Allow the user to view the conversation history and other information stored in the agent's memory
# #                                         #with st.expander("History/Memory"):
# #                                             #st.session_state.memory
# #                                             #message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
# #                     
# =============================================================================
# =============================================================================
#                            else:
# =============================================================================
  # Set up the sidebar
  #input_data_ = st.session_state.data
  # Upload manufacturing image
  #st.subheader("Default Controllable Parameters")
  # User input for the controllable independent variables
  # Display subheading for external temperature
# =============================================================================
#   input_steam_outlet_temp=143
#   input_steam_inlet_temp=153
#   input_steam_sep_line_temp=145
#   input_transfer_cool_air_temp=90
#   input_transfer_air_temp=63
#   input_ambient_temp = 90
#   ambient_humidity = 30            
#   LOD_raw_husk =  9.6
#   swell = 29.87
#   on_30 =  42.2
#   thru_70 = 0.10
#   thru_100 =  0.5
#   avg_feed_rate =  1000
#   avg_steam_pressure =  2.6
#   avg_air_velocity = 29
#   # Create a dataframe with the user input and fixed values
#   input_data_ = pd.DataFrame({
#       'Air Velocity': [float(avg_air_velocity)],
#       'Steam Inlet Temp': [float(input_steam_inlet_temp)],
#       'Steam Outlet Temp': [float(input_steam_outlet_temp)],
#       'Steam Sep Line Temp': [float(input_steam_sep_line_temp)],
#       'Transfer cool air temp':[float(input_transfer_cool_air_temp)],
#       'Steam Pressure': [float(avg_steam_pressure)],
#       'Ambient Temp': [float(input_ambient_temp)],
#       'Ambient Humidity':[float(ambient_humidity)],
#       'Feed_rate':[float(avg_feed_rate)],
#       'LOD_raw':[float(LOD_raw_husk)],
#       'Swell_vol':[float(swell)],
#       'on_30':[float(on_30)],
#       'thru_70':[float(thru_70)],
#       'thru_100':[float(thru_100)],
#       'Transfer Air':[float(input_transfer_air_temp)]
#   })
#   input_data_['const'] = 1.0 #sm.add_constant(input_data)
#   # Predict the moisture content using the trained model
#   # Display the predicted moisture content
# 
#   #st.subheader("Predicted Output %")
#   predicted_moisture_content = np.dot(input_data_, np.append(coefficients,[intercept]))[0]
# 
# 
# =============================================================================
