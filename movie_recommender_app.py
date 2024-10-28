import streamlit as st
import pandas as pd
import numpy as np
import os

st.title("Plot DNA 🧬 - File Check")

# Check all required files
files_to_check = [
    'final_movie_dataset.csv',
    'movie_similarity_matrix.npy',
    'movie_dict.pkl'
]

st.write("### Checking Required Files:")
for file in files_to_check:
    if os.path.exists(file):
        st.write(f"✅ {file} found")
        
        # Show file info
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {df.columns.tolist()}")
            st.write("- First few rows:")
            st.write(df.head())
            
        elif file.endswith('.npy'):
            matrix = np.load(file)
            st.write(f"- Shape: {matrix.shape}")
            st.write(f"- Size: {matrix.size}")
            st.write(f"- Type: {matrix.dtype}")
            
    else:
        st.error(f"❌ {file} not found")

st.write("\n### Current Directory Contents:")
st.write(os.listdir())
