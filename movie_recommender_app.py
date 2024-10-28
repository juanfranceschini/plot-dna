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
        
        # Show sample of release_date column
        st.write("Sample release dates:", df['release_date'].head())
        
        # Extract year safely
        def extract_year(date_str):
            try:
                # If it's already just a year
                if str(date_str).isdigit() and len(str(date_str)) == 4:
                    return int(date_str)
                # Try parsing as date
                return pd.to_datetime(date_str, format='mixed').year
            except:
                return None
        
        # Create movie title with year
        df['year'] = df['release_date'].apply(extract_year)
        df['movie_title_with_year'] = df.apply(
            lambda x: f"{x['movie_name']} ({x['year']})" if pd.notna(x['year']) 
            else x['movie_name'], axis=1
        )
        
        # Load similarity matrix
        similarity_matrix = np.load('movie_similarity_matrix.npy')
        
        return df, similarity_matrix
        
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.write("DataFrame info:")
        if 'df' in locals():
            st.write(df.info())
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
