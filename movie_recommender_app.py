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

@st.cache_data
def load_data():
    try:
        # Load CSV
        df = pd.read_csv('final_movie_dataset.csv')
        
        # Extract year from release_date
        df['year'] = pd.to_datetime(df['release_date']).dt.year
        
        # Create movie title with year
        df['movie_title_with_year'] = df['movie_name'] + ' (' + df['year'].astype(str) + ')'
        
        # Load similarity matrix
        similarity_matrix = np.load('movie_similarity_matrix.npy')
        
        return df, similarity_matrix
        
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        raise e

# Load data
df, similarity_matrix = load_data()

# Movie selection
st.markdown("### 🎬 Select Your Starting Movie")
try:
    start_movie = st.selectbox('', df['movie_title_with_year'].tolist(), 
                              help="Choose a movie to explore its thematic connections")
except Exception as e:
    st.error(f"Error creating movie list: {str(e)}")
    st.write("Available columns:", df.columns.tolist())
