#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Create navigation buttons
st.set_page_config(page_title="Smart Ops",layout="wide")

#streamlit app 
# Define page names
PAGE_1 =  "Yard"
PAGE_2 =  "Inbound"



nav_option = st.sidebar.radio("Smart OPs", (PAGE_1, PAGE_2 ))

# Display content based on selected page
if nav_option == PAGE_1:
    st.title('Yard')
    st.markdown("""
    <iframe title="Smart Ops - Yard" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=59c2db93-3428-4c13-beab-a6fee3eacb1e&autoAuth=true&ctid=2d2199a8-cb98-4269-b2ae-c63cf2b7c7f0" frameborder="0" allowFullScreen="true"></iframe> 
    """,unsafe_allow_html=True)
else :
    st.title('Inbound')
    st.markdown("""
    <iframe title="Smart Ops - Inbound" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=83ebd448-0b0e-4121-93b9-010bd261db9c&autoAuth=true&ctid=2d2199a8-cb98-4269-b2ae-c63cf2b7c7f0" frameborder="0" allowFullScreen="true"></iframe>
    """,unsafe_allow_html=True)


# In[ ]:




