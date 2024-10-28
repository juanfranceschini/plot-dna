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

# Step 1: Load and inspect CSV
try:
    st.write("Step 1: Loading CSV file...")
    df = pd.read_csv('final_movie_dataset.csv')
    st.write("✅ CSV loaded successfully!")
    st.write("Number of rows:", len(df))
    st.write("Number of columns:", len(df.columns))
    st.write("\nColumn names:")
    for col in df.columns:
        st.write(f"- {col}")
    st.write("\nFirst 3 rows:")
    st.write(df.head(3))

    # Step 2: Load and inspect similarity matrix
    st.write("\nStep 2: Loading similarity matrix...")
    raw_matrix = np.load('movie_similarity_matrix.npy')
    st.write("✅ Matrix loaded successfully!")
    st.write("Matrix shape:", raw_matrix.shape)
    st.write("Matrix size:", raw_matrix.size)
    
    # Step 3: Try to create movie titles
    st.write("\nStep 3: Creating movie titles...")
    # Let's try to identify the title and year columns
    title_cols = [col for col in df.columns if 'title' in col.lower()]
    year_cols = [col for col in df.columns if 'year' in col.lower()]
    
    st.write("Found title columns:", title_cols)
    st.write("Found year columns:", year_cols)
    
    if title_cols and year_cols:
        movie_titles = df[title_cols[0]].astype(str) + ' (' + df[year_cols[0]].astype(str) + ')'
        st.write("\nSample movie titles:")
        st.write(movie_titles.head())
    
except Exception as e:
    st.error(f"Error occurred: {str(e)}")
    st.write("Error location:", e.__traceback__.tb_lineno)
