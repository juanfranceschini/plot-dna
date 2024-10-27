import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load your data
print("Loading data...")
df = pd.read_csv('final_movie_dataset.csv')
original_embeddings = np.load('movie_embeddings.npy')

# Reduce embedding dimensions
print("Reducing embedding dimensions...")
pca = PCA(n_components=100)  # You can adjust this number based on your needs
reduced_embeddings = pca.fit_transform(original_embeddings)
np.save('reduced_movie_embeddings.npy', reduced_embeddings)
print("Reduced embeddings saved as 'reduced_movie_embeddings.npy'")

# Compute similarity matrix
print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(reduced_embeddings)

# Apply power transformation to spread out similarities
print("Applying power transformation...")
power = 0.2  # Reduced from 0.3 to spread out lower similarities more
similarity_matrix_powered = np.power(similarity_matrix, power)

# Apply exponential transformation to further emphasize differences
print("Applying exponential transformation...")
similarity_matrix_exp = np.exp(similarity_matrix_powered) - 1  # Subtract 1 to keep the minimum at 0

# Normalize similarity scores to 0.5-1 range
print("Normalizing similarity scores...")
scaler = MinMaxScaler(feature_range=(0.5, 1))
similarity_matrix_normalized = scaler.fit_transform(similarity_matrix_exp.flatten().reshape(-1, 1)).reshape(similarity_matrix_exp.shape)

np.save('movie_similarity_matrix.npy', similarity_matrix_normalized)
print("Normalized similarity matrix saved as 'movie_similarity_matrix.npy'")

# Create movie dictionary
print("Creating movie dictionary...")
# Ensure we have a unique identifier for each movie
if 'movie_id' not in df.columns:
    df['movie_id'] = df.index.astype(str)

# Create movie dictionary using movie_id as the key
movie_dict = df.set_index('movie_id').to_dict('index')

# Create a dictionary to map movie names to their IDs
# If there are duplicate names, we'll keep the last occurrence
movie_name_to_id = df.set_index('movie_name')['movie_id'].to_dict()

# Combine both dictionaries
combined_dict = {
    'movie_data': movie_dict,
    'movie_name_to_id': movie_name_to_id
}

# Save as a pickle file
with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(combined_dict, f)
print("Movie dictionary saved as 'movie_dict.pkl'")

print("Preprocessing complete!")
