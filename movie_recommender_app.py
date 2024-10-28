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
        
        try:
            # Show file info
            if file.endswith('.csv'):
                df = pd.read_csv(file)
                st.write(f"- Rows: {len(df)}")
                st.write(f"- Columns: {df.columns.tolist()}")
                st.write("- First few rows:")
                st.write(df.head())
                
            elif file.endswith('.npy'):
                raw_matrix = np.load(file, allow_pickle=True)
                st.write(f"- Raw size: {raw_matrix.size}")
                st.write(f"- Raw shape: {raw_matrix.shape}")
                st.write(f"- Type: {raw_matrix.dtype}")
                
                # Try to calculate square dimensions
                n = int(np.sqrt(raw_matrix.size))
                st.write(f"- Square root of size: {n}")
                if n * n == raw_matrix.size:
                    st.write(f"- Can be reshaped to ({n}, {n})")
                else:
                    st.write("- Not a perfect square matrix")
                
            elif file.endswith('.pkl'):
                import pickle
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                st.write(f"- Type: {type(data)}")
                if isinstance(data, dict):
                    st.write(f"- Number of entries: {len(data)}")
                    
        except Exception as e:
            st.error(f"Error reading {file}: {str(e)}")
            
    else:
        st.error(f"❌ {file} not found")

st.write("\n### Current Directory Contents:")
st.write(os.listdir())
