import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
import os

# Page config
st.set_page_config(
    page_title="Plot DNA 🧬",
    page_icon="🧬",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        # Load the data
        df = pd.read_csv('final_movie_dataset.csv')
        
        # Debug information
        st.write("DataFrame Info:")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("\nFirst few rows of the DataFrame:")
        st.write(df.head())
        
        # Load similarity matrix with reshape
        raw_matrix = np.load('movie_similarity_matrix.npy')
        st.write("\nRaw similarity matrix shape:", raw_matrix.shape)
        
        # Calculate the correct dimensions
        n = int(np.sqrt(raw_matrix.size))
        similarity_matrix = raw_matrix.reshape((n, n))
        st.write("Reshaped similarity matrix:", similarity_matrix.shape)
        
        return df, similarity_matrix
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load data
df, similarity_matrix = load_data()

# Movie selection
st.markdown("### 🎬 Select Your Starting Movie")
try:
    # First, let's see what columns we actually have
    st.write("Available columns:", df.columns.tolist())
    
    # Try to find the right column names
    title_column = next(col for col in df.columns if 'title' in col.lower())
    year_column = next(col for col in df.columns if 'year' in col.lower())
    
    st.write(f"Using columns: {title_column} and {year_column}")
    
    # Create movie titles
    movie_titles = df[title_column].astype(str) + ' (' + df[year_column].astype(str) + ')'
    start_movie = st.selectbox('', movie_titles.tolist(), 
                              help="Choose a movie to explore its thematic connections")
except Exception as e:
    st.error(f"Error creating movie list: {str(e)}")
    st.write("Available columns:", df.columns.tolist())
    st.write("\nFirst few rows of data:")
    st.write(df.head())
