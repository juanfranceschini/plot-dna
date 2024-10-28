import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
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
        # Load DataFrame
        df = pd.read_csv('final_movie_dataset.csv')
        
        # Create movie title with year
        df['movie_title_with_year'] = df.apply(
            lambda x: f"{x['movie_name']} ({x['release_date']})", 
            axis=1
        )
        
        # Load similarity matrix (already in correct shape)
        similarity_matrix = np.load('movie_similarity_matrix.npy')
        
        return df, similarity_matrix
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load data
df, similarity_matrix = load_data()

# Movie selection
st.markdown("### 🎬 Select Your Starting Movie")
try:
    movie_titles = df['movie_title_with_year'].tolist()
    start_movie = st.selectbox('', movie_titles, 
                              help="Choose a movie to explore its thematic connections")
    
    # Debug info
    st.write("Data loaded successfully:")
    st.write(f"- Number of movies: {len(df)}")
    st.write(f"- Matrix shape: {similarity_matrix.shape}")
    st.write(f"- Sample titles: {movie_titles[:5]}")
    
except Exception as e:
    st.error(f"Error creating movie list: {str(e)}")
    st.write("Available columns:", df.columns.tolist())
