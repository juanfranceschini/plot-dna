import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Page config
st.set_page_config(
    page_title="Plot DNA 🧬",
    page_icon="🧬",
    layout="wide"
)

st.title("Plot DNA 🧬 - Matrix Generation")

try:
    # 1. Load the movie embeddings
    st.write("Loading embeddings...")
    embeddings = np.load('reduced_movie_embeddings.npy')
    st.write(f"Embeddings shape: {embeddings.shape}")
    
    # 2. Load the DataFrame
    st.write("\nLoading DataFrame...")
    df = pd.read_csv('final_movie_dataset.csv')
    st.write(f"DataFrame shape: {df.shape}")
    
    # 3. Generate similarity matrix
    st.write("\nGenerating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    st.write(f"Generated matrix shape: {similarity_matrix.shape}")
    
    # 4. Save the new matrix
    st.write("\nSaving matrix...")
    np.save('movie_similarity_matrix.npy', similarity_matrix)
    st.write("✅ Matrix saved successfully!")
    
    # 5. Verify the saved matrix
    st.write("\nVerifying saved matrix...")
    loaded_matrix = np.load('movie_similarity_matrix.npy')
    st.write(f"Loaded matrix shape: {loaded_matrix.shape}")
    
    if loaded_matrix.shape == (len(df), len(df)):
        st.success("Matrix dimensions match DataFrame!")
    else:
        st.error("Matrix dimensions don't match DataFrame!")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("\nCurrent directory contents:")
    st.write(os.listdir())

# Load data
try:
    df, similarity_matrix = load_data()
    st.write("✅ Data loaded successfully!")
    st.write(f"Number of movies: {len(df)}")
    st.write(f"Matrix shape: {similarity_matrix.shape}")
    
    # Movie selection
    st.markdown("### 🎬 Select Your Starting Movie")
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
        
        # Load similarity matrix
        similarity_matrix = np.load('movie_similarity_matrix.npy')
        
        # Verify shapes
        if similarity_matrix.shape != (len(df), len(df)):
            raise ValueError(f"Matrix shape {similarity_matrix.shape} doesn't match DataFrame length {len(df)}")
        
        return df, similarity_matrix.astype(np.float32)  # Convert to float32 for memory efficiency
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e
