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