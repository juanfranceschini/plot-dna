import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random

# Page config
st.set_page_config(
    page_title="Plot DNA 🧬",
    page_icon="🧬",
    layout="wide"
)

# Title and intro
st.title("Plot DNA 🧬")
st.markdown("""
    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <h3 style='margin-top: 0'>Discover Movies Through Story Patterns</h3>
        <p>Plot DNA analyzes movie plots to find thematic connections, revealing unexpected similarities between films. 
        Unlike traditional recommenders that rely on genres or ratings, this tool discovers movies that share narrative DNA.</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('final_movie_dataset.csv')
    similarity_matrix = np.load('movie_similarity_matrix.npy')
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
    
    st.write("Similarity Matrix Shape:", similarity_matrix.shape)
    st.write("Similarity Matrix Range:", np.min(similarity_matrix), "to", np.max(similarity_matrix))
    
    return df, similarity_matrix

def convert_to_native(value):
    if isinstance(value, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(value)
    elif isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    elif np.isnan(value):
        return 0.0
    return value

def create_movie_path_graph(df, similarity_matrix, start_idx, max_depth=2, max_connections=4):
    G = nx.Graph()
    
    def add_node_with_metadata(idx, depth):
        if not G.has_node(idx):
            similarity = convert_to_native(similarity_matrix[start_idx, idx])
            G.add_node(idx, 
                      title=df.loc[convert_to_native(idx), 'movie_title_with_year'],
                      similarity=similarity,
                      depth=convert_to_native(depth))
    
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
            similarities = similarity_matrix[current_idx]
            most_similar = np.argsort(similarities)[-max_connections-1:-1]
            
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
    
    return G

def create_pyvis_network(graph, df, start_movie):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    
    # Get similarity range for color mapping
    similarities = [data['similarity'] for _, data in graph.nodes(data=True)]
    min_sim = min(similarities) if similarities else 0
    max_sim = max(similarities) if similarities else 1
    
    # Prevent division by zero
    if max_sim == min_sim:
        max_sim = min_sim + 1
    
    # Create color map
    color_map = plt.cm.RdPu
    
    # Add nodes
    for node in graph.nodes(data=True):
        node_id, node_data = node
        similarity = node_data.get('similarity', 0.0)
        if np.isnan(similarity):
            similarity = 0.0
        
        if max_sim > min_sim:
            norm_similarity = (similarity - min_sim) / (max_sim - min_sim)
        else:
            norm_similarity = 0.5
            
        rgba_color = color_map(norm_similarity)
        hex_color = mcolors.to_hex(rgba_color)
        
        size = 15 + (norm_similarity * 25) if not np.isnan(norm_similarity) else 15
        
        net.add_node(str(node_id), 
                     label=node_data['title'], 
                     title=f"{node_data['title']}|Similarity: {similarity:.2f}|Depth: {node_data['depth']}",
                     color=hex_color,
                     size=size,
                     borderWidth=2,
                     borderWidthSelected=4,
                     borderColor='#000000')
    
    # Add edges
    for edge in graph.edges(data=True):
        weight = edge[2].get('weight', 0.0)
        if np.isnan(weight):
            weight = 0.0
        net.add_edge(str(edge[0]), str(edge[1]), value=weight, title=f"Similarity: {weight:.2f}")
    
    # Set options
    net.set_options("""
    const options = {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springStrength": 0.05,
                "damping": 0.09,
                "avoidOverlap": 0
            },
            "stabilization": {
                "iterations": 100
            }
        },
        "interaction": {
            "navigationButtons": true,
            "zoomView": true
        }
    }
    """)
    
    # Save and load the network
    net.save_graph("movie_network.html")
    
    # Add custom styling
    with open("movie_network.html", "r", encoding="utf-8") as file:
        content = file.read()
    
    # Add font imports and styling
    font_import = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
    <style>
    canvas {
        font-family: 'Bebas Neue', sans-serif !important;
        font-weight: normal !important;
    }
    .vis-network, .vis-network canvas, .vis-network div {
        font-family: 'Bebas Neue', sans-serif !important;
        font-weight: normal !important;
    }
    </style>
    """
    content = content.replace("</head>", f"{font_import}</head>")
    
    # Add the physics control JavaScript
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
    </script>
    """
    content = content.replace("</body>", f"{custom_js}</body>")
    
    with open("movie_network.html", "w", encoding="utf-8") as file:
        file.write(content)

# Load data
df, similarity_matrix = load_data()

# Create tabs
tab1, tab2 = st.tabs(["Movie Network", "How It Works"])

with tab1:
    # Movie selection
    st.markdown("### 🎬 Select Your Starting Movie")
    start_movie = st.selectbox('', df['movie_title_with_year'].tolist(), 
                            help="Choose a movie to explore its thematic connections")

    # Parameters and buttons
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider('Exploration Depth', 1, 3, 2, 
                            help="How far to explore from your selected movie")
    with col2:
        max_connections = st.slider('Similar Movies per Level', 1, 6, 4, 
                                help="Number of similar movies to show for each film")

    col3, col4 = st.columns(2)
    generate_button = col3.button('Generate Graph')
    recommend_button = col4.button('Recommend a Movie')

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
                    st.components.v1.html(html_string, height=600)
                else:
                    st.warning("No similar movies found. Try adjusting the parameters.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your data and parameters, and try again.")

    # Display existing graph if it exists
    elif 'graph' in st.session_state:
        create_pyvis_network(st.session_state['graph'], df, start_movie)
        with open("movie_network.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600)

    # Recommendation logic
    if recommend_button:
        if 'graph' not in st.session_state:
            st.warning("Please generate the graph first!")
        else:
            random_movie_index = random.choice([n for n in st.session_state['graph'].nodes() 
                                              if n != st.session_state['selected_movie_index']])
            movie_data = df.loc[random_movie_index]
            similarity = st.session_state['graph'].nodes[random_movie_index]['similarity']
            
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

# Footer
st.markdown(f"""
    <div style='margin-top: 50px; padding: 20px; text-align: center; font-size: 14px; color: #666;'>
        Plot DNA analyzes {len(df):,} movies to find thematic connections. 
        Built with Streamlit and PyVis. 
        <a href="https://www.linkedin.com/in/juan-franceschini-uy/" target="_blank" style="color: #666; text-decoration: underline; margin: 0 10px;">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
