import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Plot DNA 🧬",
    page_icon="🧬",
    layout="wide"
)

st.title("Plot DNA 🧬")

# Basic data loading
try:
    # Load CSV
    st.write("Attempting to load CSV...")
    df = pd.read_csv('final_movie_dataset.csv')
    st.write("CSV loaded successfully!")
    
    # Show basic info
    st.write("DataFrame shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("\nFirst few rows:")
    st.write(df.head())
    
    # Load numpy array
    st.write("\nAttempting to load similarity matrix...")
    raw_matrix = np.load('movie_similarity_matrix.npy')
    st.write("Matrix loaded successfully!")
    st.write("Matrix shape:", raw_matrix.shape)
    
except Exception as e:
    st.error(f"Error occurred: {str(e)}")
    st.write("Current working directory contents:")
    import os
    st.write(os.listdir())
