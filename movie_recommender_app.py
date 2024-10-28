import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import colorsys
import pickle
import html
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Must be the first Streamlit command
st.set_page_config(
    page_title="Plot DNA 🧬",
    page_icon="🧬",
    layout="wide"
)

def parse_date_and_extract_year(date_string):
    if pd.isna(date_string):
        return None
    try:
        # Try parsing as full date
        date = pd.to_datetime(date_string)
        return date.year
    except:
        # If full date parsing fails, try extracting year directly
        import re
        year_match = re.search(r'\d{4}', str(date_string))
        if year_match:
            return int(year_match.group())
        else:
            return None

def create_movie_title_with_year(row):
    year = parse_date_and_extract_year(row['release_date'])
    if year:
        return f"{row['movie_name']} ({year})"
    else:
        return row['movie_name']

@st.cache_data
def load_data():
    df = pd.read_csv('final_movie_dataset.csv')
    
    # Load similarity matrix with error handling
    similarity_matrix = np.load('movie_similarity_matrix.npy')
    
    # Replace NaN values with zeros
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
    
    # Add diagnostic information
    st.write("Similarity Matrix Shape:", similarity_matrix.shape)
    st.write("Similarity Matrix Range:", np.min(similarity_matrix), "to", np.max(similarity_matrix))
    
    # Verify the matrix has valid values
    if np.all(np.isnan(similarity_matrix)):
        st.error("Error: Similarity matrix contains all NaN values!")
        similarity_matrix = np.zeros((len(df), len(df)))  # Fallback to zeros
    
    return df, similarity_matrix

def create_movie_path_graph(df, similarity_matrix, start_idx, max_depth=2, max_connections=4):
    G = nx.Graph()
    
    # Convert numpy types to native Python types
    def convert_to_native(value):
        if isinstance(value, (np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(value)
        elif isinstance(value, (np.float16, np.float32, np.float64)):
            return float(value)
        elif np.isnan(value):
            return 0.0
        return value

    def add_node_with_metadata(idx, depth):
        if not G.has_node(idx):
            similarity = float(similarity_matrix[start_idx, idx])
            st.write(f"Raw similarity value for {idx}: {similarity}")
            G.add_node(idx, 
                      title=df.loc[convert_to_native(idx), 'movie_title_with_year'],
                      similarity=similarity,
                      depth=convert_to_native(depth))
            st.write(f"Added node: {df.loc[convert_to_native(idx), 'movie_title_with_year']} (similarity: {similarity:.2f}, depth: {depth})")
    
    # Add start node
    add_node_with_metadata(start_idx, 0)
    
    # BFS to add similar movies
    current_depth = 0
    nodes_to_explore = [(start_idx, 0)]
    explored_nodes = set([start_idx])
    
    while nodes_to_explore and current_depth < max_depth:
        current_idx, depth = nodes_to_explore.pop(0)
        
        if depth > current_depth:
            current_depth = depth
            
        if depth < max_depth:
            # Get similar movies
            similarities = similarity_matrix[current_idx]
            most_similar = np.argsort(similarities)[-max_connections-1:-1]
            
            # Debug similar movies
            st.write(f"\nFinding similar movies for index {current_idx}")
            st.write(f"Number of similar movies found: {len(most_similar)}")
            
            for similar_idx in most_similar:
                similar_idx = convert_to_native(similar_idx)
                if similar_idx not in explored_nodes:
                    add_node_with_metadata(similar_idx, depth + 1)
                    similarity = convert_to_native(similarities[similar_idx])
                    G.add_edge(current_idx, similar_idx, weight=similarity)
                    nodes_to_explore.append((similar_idx, depth + 1))
                    explored_nodes.add(similar_idx)
                elif not G.has_edge(current_idx, similar_idx):
                    similarity = convert_to_native(similarities[similar_idx])
                    G.add_edge(current_idx, similar_idx, weight=similarity)
    
    # Final graph statistics
    st.write("\nFinal Graph Statistics:")
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    
    return G

def create_pyvis_network(graph, df, start_movie):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    
    # Get similarity range for color mapping
    similarities = [data['similarity'] for _, data in graph.nodes(data=True)]
    min_sim = min(similarities) if similarities else 0
    max_sim = max(similarities) if similarities else 1
    
    # Prevent division by zero
    if max_sim == min_sim:
        max_sim = min_sim + 1  # Add 1 to prevent division by zero
    
    # Create color map
    color_map = plt.cm.RdPu
    
    # Add nodes
    for node in graph.nodes(data=True):
        node_id, node_data = node
        similarity = node_data.get('similarity', 0.0)
        if np.isnan(similarity):
            similarity = 0.0
        
        # Prevent division by zero in normalization
        if max_sim > min_sim:
            norm_similarity = (similarity - min_sim) / (max_sim - min_sim)
        else:
            norm_similarity = 0.5  # Default to middle value if no range
            
        rgba_color = color_map(norm_similarity)
        hex_color = mcolors.to_hex(rgba_color)
        
        # Base size on similarity with safety check
        size = 15 + (norm_similarity * 25) if not np.isnan(norm_similarity) else 15
        
        net.add_node(str(node_id), 
                     label=node_data['title'], 
                     title=f"{node_data['title']}|Similarity: {similarity:.2f}|Depth: {node_data['depth']}",
                     color=hex_color,
                     size=size,
                     borderWidth=2,
                     borderWidthSelected=4,
                     borderColor='#000000')
    
    # Add edges with error handling
    for edge in graph.edges(data=True):
        weight = edge[2].get('weight', 0.0)
        if np.isnan(weight):
            weight = 0.0
        net.add_edge(str(edge[0]), str(edge[1]), value=weight, title=f"Similarity: {weight:.2f}")
    
    # Save and modify the network
    net.save_graph("movie_network.html")
    
    # Modify the HTML to force Bebas Neue font
    with open("movie_network.html", "r", encoding="utf-8") as file:
        content = file.read()
    
    # Add font imports and styling
    font_import = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
    <style>
    /* Force Bebas Neue on all network labels with normal weight */
    canvas {
        font-family: 'Bebas Neue', sans-serif !important;
        font-weight: normal !important;
    }
    .vis-network, .vis-network canvas, .vis-network div {
        font-family: 'Bebas Neue', sans-serif !important;
        font-weight: normal !important;
    }
    </style>
    <script>
    // Add font to vis.js options
    var options = {
        nodes: {
            font: {
                face: 'Bebas Neue',
                size: 18,
                bold: false
            }
        }
    };
    </script>
    """
    content = content.replace("</head>", f"{font_import}</head>")
    
    # Add the physics control JavaScript with normal weight font
    custom_js = """
    <script>
    network.on("stabilizationIterationsDone", function() {
        network.setOptions({ physics: false });
    });
    
    network.on("dragEnd", function() {
        network.setOptions({ physics: true });
        setTimeout(function() {
            network.setOptions({ physics: false });
        }, 1000);
    });
    
    // Force font update after network is ready
    network.on("afterDrawing", function() {
        network.setOptions({
            nodes: {
                font: {
                    face: 'Bebas Neue',
                    size: 18,
                    bold: false
                }
            }
        });
    });
    </script>
    """
    content = content.replace("</body>", f"{custom_js}</body>")
    
    with open("movie_network.html", "w", encoding="utf-8") as file:
        file.write(content)

def get_random_path(G, start_node, path_length=5):
    path = [start_node]
    current_node = start_node
    visited = set([start_node])
    for _ in range(path_length - 1):
        neighbors = [n for n in G.neighbors(current_node) if n not in visited]
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node
    return path

def display_movie_card(movie_data, step):
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <h2>Step {step}: {movie_data['movie_title_with_year']}</h2>
            <p><strong>Genres:</strong> {movie_data.get('genres', 'N/A')}</p>
            <p><strong>Release Date:</strong> {movie_data.get('release_date', 'N/A')}</p>
            <p><strong>Box Office:</strong> ${movie_data.get('box_office_revenue', 0):,}</p>
            <p><strong>Plot:</strong> {movie_data.get('plot_summary', 'N/A')[:200]}...</p>
        </div>
        """, unsafe_allow_html=True)

def get_next_movie_recommendation(G):
    # Choose a random node from the graph
    return random.choice(list(G.nodes()))

def estimate_node_count(max_depth, max_connections):
    total = 1
    for depth in range(1, max_depth + 1):
        total += max_connections ** depth
    return total

def create_recommendation_card(movie_data, similarity):
    st.markdown("""
    <style>
    .recommendation-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6e9ff 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 1px solid #e1e4e8;
    }
    .movie-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 32px;
        color: #2c3e50;
        margin-bottom: 20px;
        border-bottom: 2px solid #b3c6ff;
        padding-bottom: 10px;
    }
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .metadata-item {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Your existing recommendation card code with new styling

# Load data
df, embeddings, movie_dict, movie_name_to_id, wiki_id_to_index = load_data()

# Load pre-computed similarity matrix
similarity_matrix = np.load('movie_similarity_matrix.npy')

# Add this near where you load the data
st.write("Similarity Matrix Shape:", similarity_matrix.shape)
st.write("Similarity Matrix Range:", np.min(similarity_matrix), "to", np.max(similarity_matrix))

# Custom CSS for an elegant, cohesive design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4A4A8F;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #5D5DA8;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stSelectbox label, .stSlider label {
        color: #34495e;
        font-weight: bold;
    }
    .movie-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .movie-card h2 {
        color: #2c3e50;
    }
    .movie-card p {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for parameters and instructions
st.sidebar.header('How it Works')
st.sidebar.write("""
Plot DNA analyzes movie plots to find thematic connections between films. 
Unlike traditional recommenders that rely on genres or ratings, 
this tool discovers movies that share similar narrative elements and story patterns.
""")

st.sidebar.markdown("""
<style>
    .sidebar-header {
        font-family: 'Bebas Neue', sans-serif;
        color: #2c3e50;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .parameter-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-header">Graph Controls</div>', unsafe_allow_html=True)
with st.sidebar.container():
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    max_depth = st.slider('Exploration Depth', 1, 3, 2, 
                         help="How far to explore from your selected movie")
    max_connections = st.slider('Connection Strength', 1, 6, 4, 
                              help="How many similar movies to show for each film")
    st.markdown('</div>', unsafe_allow_html=True)

# Calculate estimated nodes but don't display
estimated_nodes = estimate_node_count(max_depth, max_connections)

# Only show warning if the graph might be too large
if estimated_nodes > 400:
    st.sidebar.warning("Warning: Large number of nodes may slow down rendering.")

# Add this near the top of your app, after loading the DataFrame
total_movies = len(df)
st.sidebar.write(f"Total movies in database: {total_movies:,}")

# Title and intro
st.title("Plot DNA 🧬")
st.markdown("""
    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <h3 style='margin-top: 0'>Discover Movies Through Story Patterns</h3>
        <p>Plot DNA analyzes movie plots to find thematic connections, revealing unexpected similarities between films. 
        Unlike traditional recommenders that rely on genres or ratings, this tool discovers movies that share narrative DNA.</p>
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Movie Network", "How It Works"])

with tab1:
    # Movie selection
    st.markdown("### 🎬 Select Your Starting Movie")
    start_movie = st.selectbox('', df['movie_title_with_year'].tolist(), 
                            help="Choose a movie to explore its thematic connections")

    # Parameters and buttons
    col1, col2 = st.columns(2)
    generate_button = col1.button('Generate Graph', key='generate_button')
    recommend_button = col2.button('Recommend a Movie', key='recommend_button')

    # Graph generation
    if generate_button:
        with st.spinner('Generating graph...'):
            try:
                selected_movie_index = df[df['movie_title_with_year'] == start_movie].index[0]
                
                G = create_movie_path_graph(df, similarity_matrix, selected_movie_index, 
                                          max_depth=max_depth, 
                                          max_connections=max_connections)
                
                st.session_state['graph'] = G
                st.session_state['selected_movie_index'] = selected_movie_index

                if G.number_of_nodes() > 1:
                    create_pyvis_network(G, df, start_movie)
                    with open("movie_network.html", 'r', encoding='utf-8') as f:
                        html_string = f.read()
                    st.components.v1.html(html_string, height=600, width=None)
                else:
                    st.warning("No similar movies found. Try adjusting the parameters or selecting a different movie.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your data and parameters, and try again.")

    # Display existing graph if it exists
    elif 'graph' in st.session_state:
        create_pyvis_network(st.session_state['graph'], df, start_movie)
        with open("movie_network.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600, width=None)

    # Recommendation logic
    if recommend_button:
        if 'graph' not in st.session_state:
            st.warning("Please generate the graph first!")
        else:
            # Get random movie from graph
            G = st.session_state['graph']
            selected_movie_index = st.session_state['selected_movie_index']
            nodes = list(G.nodes())
            nodes.remove(selected_movie_index)
            if nodes:
                random_movie_index = random.choice(nodes)
                movie_data = df.loc[random_movie_index]
                similarity = G.nodes[random_movie_index]['similarity']
                
                # Create the recommendation card
                st.markdown("""
                <style>
                .movie-card {
                    background: linear-gradient(135deg, #f0f8ff 0%, #e6e9ff 100%);
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin: 20px 0;
                    border: 1px solid #e1e4e8;
                }
                .movie-title {
                    font-family: 'Bebas Neue', sans-serif;
                    font-size: 32px;
                    color: #2c3e50;
                    margin-bottom: 20px;
                }
                .movie-metadata {
                    font-size: 16px;
                    color: #34495e;
                    margin: 5px 0;
                }
                .metadata-label {
                    font-weight: bold;
                    color: #2c3e50;
                }
                .similarity-score {
                    font-size: 18px;
                    color: #34495e;
                    margin-top: 15px;
                    padding-top: 15px;
                    border-top: 2px solid #b3c6ff;
                }
                </style>
                """, unsafe_allow_html=True)

                # Format metadata
                release_date = pd.to_datetime(movie_data['release_date']).strftime('%B %d, %Y') if pd.notna(movie_data['release_date']) else 'Unknown'
                runtime = f"{movie_data['runtime']} minutes" if pd.notna(movie_data['runtime']) else 'Unknown'
                genres = movie_data['genres'].replace('[', '').replace(']', '').replace("'", "") if pd.notna(movie_data['genres']) else 'Unknown'

                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">Here is your next movie. Enjoy!</div>
                    <div class="movie-title">{movie_data['movie_title_with_year']}</div>
                    <div class="movie-metadata">
                        <span class="metadata-label">Release Date:</span> {release_date}
                    </div>
                    <div class="movie-metadata">
                        <span class="metadata-label">Runtime:</span> {runtime}
                    </div>
                    <div class="movie-metadata">
                        <span class="metadata-label">Genres:</span> {genres}
                    </div>
                    <div class="similarity-score">Thematic Similarity: {similarity:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### Understanding the Visualization
    - **Node Size**: Larger nodes indicate stronger thematic connections
    - **Colors**: Darker purple indicates higher similarity
    - **Connections**: Lines show direct plot similarities
    
    ### Tips
    - 🔍 Use the navigation controls to zoom and pan
    - 🎯 Click and drag nodes to explore connections
    - 🎲 Use 'Recommend a Movie' for surprising discoveries
    
    ### About the Algorithm
    Plot DNA uses natural language processing to analyze movie plots and find hidden 
    connections between films. The similarity score (0-1) indicates how closely the 
    narrative elements match between movies.
    """)

# Footer with GitHub and LinkedIn links
st.markdown(f"""
    <div style='margin-top: 50px; padding: 20px; text-align: center; font-size: 14px; color: #666;'>
        Plot DNA analyzes {len(df):,} movies to find thematic connections. 
        Built with Streamlit and PyVis. 
        <a href="https://github.com/juanfranceschini" target="_blank" style="color: #666; text-decoration: underline; margin: 0 10px;">GitHub</a> | 
        <a href="https://www.linkedin.com/in/juan-franceschini-uy/" target="_blank" style="color: #666; text-decoration: underline; margin: 0 10px;">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)






