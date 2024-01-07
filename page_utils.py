from PIL import Image
import streamlit as st
from streamlit.components.v1 import html

def display_page_header(title):
    col1, col2, col3 = st.columns([1,4,1])

    # Display content in each column
    with col1:
        st.image('fractal.png')

    with col2:
        st.markdown(f"<h2 style='text-align: center;font-size: 40px;font-weight: bold;line-height:2px;'>{title}</h2>", unsafe_allow_html=True)