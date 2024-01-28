"""

All code contributed to Wordlit.net is © 2024 by Sahir Maharaj. 
The content is licensed under the Creative Commons Attribution 4.0 International License. 
This allows for sharing and adaptation, provided appropriate credit is given, and any changes made are indicated.

When using the code from Wordlit.net, please credit as follows: "Code sourced from Wordlit.net, authored by Sahir Maharaj, 2024."

For reporting bugs, requesting features, or further inquiries, please reach out to Sahir Maharaj at sahir@sahirmaharaj.com.

Connect with Sahir Maharaj on LinkedIn for updates and potential collaborations: https://www.linkedin.com/in/sahir-maharaj/

"""

import spacy
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import time
import docx
import pdfplumber
import requests
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

def extract_text_from_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except Exception as e:
        return f"Error fetching content from URL: {str(e)}"

def extract_entities_with_ner_model(text):
    doc = nlp(text[:6000])
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_entity_pairs(doc):
    entity_pairs = [
        (subj.text, obj.text)
        for token in doc
        if token.dep_ in {"ROOT", "conj", "relcl", "xcomp", "ccomp", "acomp"}
        for subj in (w for w in token.lefts if w.dep_ in {"subj", "nsubj", "nsubjpass"})
        for obj in (
            w for w in token.rights if w.dep_ in {"dobj", "attr", "prep", "pobj"}
        )
    ]
    return entity_pairs


def build_knowledge_graph(text):
    start_time = time.time()
    doc = nlp(text)
    entities = extract_entities_with_ner_model(text)
    entity_pairs = extract_entity_pairs(doc)

    G = nx.Graph()
    G.add_nodes_from((word, {"label": label}) for word, label in entities)
    G.add_edges_from(entity_pairs)

    processing_time = time.time() - start_time
    return G, processing_time


def create_traces(
    G,
    pos,
    color_scheme,
    show_labels,
    node_size,
    edge_thickness,
    show_labels_on_hover,
    edge_color,
    node_shape,
):
    edge_x, edge_y, node_x, node_y, node_degrees, text = [], [], [], [], [], []

    for edge in G.edges():
        x0, y0, x1, y1 = *pos[edge[0]], *pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    hover_text = (
        [f"{node} ({G.nodes[node].get('label', 'NA')})" for node in G.nodes()]
        if show_labels_on_hover
        else ["" for _ in G.nodes()]
    )

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_degrees.append(G.degree(node))
        if show_labels:
            text.append(f"{node} ({G.nodes[node].get('label', 'NA')})")
        else:
            text.append("")

    actual_edge_color = edge_color if edge_color else "#888"

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=edge_thickness, color=actual_edge_color),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        hoverinfo="text" if show_labels_on_hover else "none",
        text=hover_text if show_labels_on_hover else text,
        marker=dict(
            symbol=node_shape,
            showscale=True,
            colorscale=color_scheme,
            size=node_size,
            color=node_degrees,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    return edge_trace, node_trace


def plot_graph(
    G,
    layout_spacing,
    color_scheme,
    show_labels,
    node_size,
    edge_thickness,
    show_labels_on_hover,
    edge_color,
    node_shape,
):
    pos = nx.spring_layout(G, k=layout_spacing)
    edge_trace, node_trace = create_traces(
        G,
        pos,
        color_scheme,
        show_labels,
        node_size,
        edge_thickness,
        show_labels_on_hover,
        edge_color,
        node_shape,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Knowledge Graph Visualization",
            titlefont_size=20,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Knowledge graph constructed by Sahir Maharaj using NLP and NetworkX",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        ),
    )

    return fig


def calculate_graph_statistics(G):

    stats = {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Graph Density": nx.density(G),
        "Connected Components": nx.number_connected_components(G),
        "Assortativity Coefficient": nx.degree_assortativity_coefficient(G),
    }

    degrees = pd.Series({node: G.degree(node) for node in G.nodes()})
    if not degrees.empty:
        stats["Average Degree"] = degrees.mean()
        stats["Maximum Degree"] = degrees.max()
        stats["Minimum Degree"] = degrees.min()
        degree_distribution = degrees.value_counts().sort_index()

    top_nodes = degrees.nlargest(5).to_frame("Connections")
    top_nodes["Word Length"] = top_nodes.index.map(lambda x: len(x))

    centrality = nx.degree_centrality(G)
    top_centrality = pd.Series(centrality).nlargest(5).to_frame("Centrality")

    node_types = pd.Series(
        [data.get("label", "Unknown") for _, data in G.nodes(data=True)]
    )
    edge_types = pd.Series([G[u][v].get("type", "Unknown") for u, v in G.edges()])

    node_types = pd.Series(
        [data.get("label", "Unknown") for _, data in G.nodes(data=True)]
    )

    num_components = nx.number_connected_components(G)
    components_color = "green" if num_components == 1 else "red"

    assortativity_coefficient = nx.degree_assortativity_coefficient(G)

    num_components = nx.number_connected_components(G)

    clustering_coefficient = nx.average_clustering(G)

    betweenness_centrality = nx.betweenness_centrality(G)
    top_betweenness_centrality = (
        pd.Series(betweenness_centrality).nlargest(5).to_frame("Betweenness Centrality")
    )

    closeness_centrality = nx.closeness_centrality(G)
    top_closeness_centrality = (
        pd.Series(closeness_centrality).nlargest(5).to_frame("Closeness Centrality")
    )

    stats["Connected Components"] = nx.number_connected_components(G)
    stats["Largest Component Size"] = len(max(nx.connected_components(G), key=len))
    stats["Average Clustering Coefficient"] = nx.average_clustering(G)

    self_loops = sum(1 for node in G.nodes() if G.has_edge(node, node))
    stats["Number of Self-loops"] = self_loops

    stats["Number of Triangles"] = sum(nx.triangles(G).values()) // 3
    stats["Assortativity Coefficient"] = nx.degree_assortativity_coefficient(G)

    pagerank = nx.pagerank(G)
    top_pagerank = pd.Series(pagerank).nlargest(5).to_frame("PageRank")

    if nx.is_connected(G):
        eccentricity = nx.eccentricity(G)
        stats["Average Eccentricity"] = sum(eccentricity.values()) / len(eccentricity)

    node_labels = [data.get("label", "Unknown") for _, data in G.nodes(data=True)]
    node_label_counts = pd.Series(node_labels).value_counts()

    stats["Most Common Entity Types"] = node_label_counts.idxmax()

    avg_word_length = sum(len(node) for node in G.nodes()) / len(G.nodes())
    stats["Average Word Length of Nodes"] = avg_word_length

    stats["Number of Unique Entity Types"] = len(set(node_labels))

    edge_counts = pd.Series([edge for edge in G.edges()]).value_counts()
    stats["Most Common Edges"] = edge_counts.idxmax() if not edge_counts.empty else None

    stats["Edge to Node Ratio"] = len(G.edges()) / len(G.nodes())

    isolated_nodes = list(nx.isolates(G))
    stats["Isolated Node Percentage"] = len(isolated_nodes) / len(G.nodes())

    entity_counts = pd.Series([node for node in G.nodes()]).value_counts()
    stats["Top Repeated Entities"] = entity_counts.idxmax()

    stats["Graph Compactness"] = nx.number_connected_components(G) / len(G.nodes())

    stats["Connected Components"] = nx.number_connected_components(G)
    stats["Largest Component Size"] = len(max(nx.connected_components(G), key=len))
    stats["Average Clustering Coefficient"] = nx.average_clustering(G)
    self_loops = sum(1 for node in G.nodes() if G.has_edge(node, node))
    stats["Number of Self-loops"] = self_loops
    stats["Number of Triangles"] = sum(nx.triangles(G).values()) // 3
    stats["Assortativity Coefficient"] = nx.degree_assortativity_coefficient(G)
    pagerank = nx.pagerank(G)
    top_pagerank = pd.Series(pagerank).nlargest(5).to_frame("PageRank")

    stats["Global Efficiency"] = nx.global_efficiency(G)

    stats["Local Efficiency"] = nx.local_efficiency(G)

    stats["Transitivity"] = nx.transitivity(G)

    if nx.is_connected(G):
        stats["Wiener Index"] = nx.wiener_index(G)

    for key, value in stats.items():
        if isinstance(value, float):
            stats[key] = round(value, 2)

    return (
        stats,
        top_nodes,
        degree_distribution,
        top_centrality,
        node_label_counts,
        num_components,
        components_color,
        assortativity_coefficient,
        top_betweenness_centrality,
        top_closeness_centrality,
        top_pagerank,
    )


def calculate_text_statistics(text):
    text_stats = {
        "Total Number of Characters": len(text),
        "Total Number of Tokens": len(text.split()),
        "Total Number of Sentences": text.count(".")
        + text.count("!")
        + text.count("?"),
        "Average Token Length": sum(len(word) for word in text.split())
        / len(text.split())
        if text.split()
        else 0,
        "Number of Unique Tokens": len(set(text.split())),
        "Number of Paragraphs": text.count("\n") + 1,
        "Number of Commas": text.count(","),
        "Number of Exclamation Marks": text.count("!"),
        "Number of Question Marks": text.count("?"),
        "Average Sentence Length": len(text.split())
        / (text.count(".") + text.count("!") + text.count("?"))
        if text.count(".") + text.count("!") + text.count("?") > 0
        else 0,
    }

    for key, value in text_stats.items():
        if isinstance(value, float):
            text_stats[key] = round(value, 2)

    return text_stats


def safe_metric_display(label, value):

    if not isinstance(value, (int, float, str, type(None))):
        value = str(value)
    st.metric(label, value)


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])


def main():

    _author_ = "https://www.linkedin.com/in/sahir-maharaj/"

    st.title("Wordlit.net")
    st.markdown(
        """
        This tool uses NLP to generate a knowledge graph from the text you provide.\n
        Enter any text or upload a file and hit the '**Generate**' button to visualize the connections between entities.
        
        All code contributed by [Sahir Maharaj](%s) is licensed under Attribution 4.0 International
        """
        % _author_
    )

    with st.sidebar:
        st.header("Customize Graph")
        with st.form("graph_parameters"):
            layout_spacing = st.slider("Layout Spacing", 0.1, 1.0, 0.5)
            color_scheme = st.selectbox(
                "Color Scheme",
                [
                    "aggrnyl",
                    "agsunset",
                    "algae",
                    "amp",
                    "armyrose",
                    "balance",
                    "blackbody",
                    "bluered",
                    "blues",
                    "blugrn",
                    "bluyl",
                    "brbg",
                    "brwnyl",
                    "bugn",
                    "bupu",
                    "burg",
                    "burgyl",
                    "cividis",
                    "curl",
                    "darkmint",
                    "deep",
                    "delta",
                    "dense",
                    "earth",
                    "edge",
                    "electric",
                    "emrld",
                    "fall",
                    "geyser",
                    "gnbu",
                    "gray",
                    "greens",
                    "greys",
                    "haline",
                    "hot",
                    "hsv",
                    "ice",
                    "icefire",
                    "inferno",
                    "jet",
                    "magenta",
                    "magma",
                    "matter",
                    "mint",
                    "mrybm",
                    "mygbm",
                    "oranges",
                    "orrd",
                    "oryel",
                    "oxy",
                    "peach",
                    "phase",
                    "picnic",
                    "pinkyl",
                    "piyg",
                    "plasma",
                    "plotly3",
                    "portland",
                    "prgn",
                    "pubu",
                    "pubugn",
                    "puor",
                    "purd",
                    "purp",
                    "purples",
                    "purpor",
                    "rainbow",
                    "rdbu",
                    "rdgy",
                    "rdpu",
                    "rdylbu",
                    "rdylgn",
                    "redor",
                    "reds",
                    "solar",
                    "spectral",
                    "speed",
                    "sunset",
                    "sunsetdark",
                    "teal",
                    "tealgrn",
                    "tealrose",
                    "tempo",
                    "temps",
                    "thermal",
                    "tropic",
                    "turbid",
                    "turbo",
                    "twilight",
                    "viridis",
                    "ylgn",
                    "ylgnbu",
                    "ylorbr",
                    "ylorrd",
                ],
                index=91,
            )
            show_labels = st.checkbox("Show Node Labels", value=True)
            show_labels_on_hover = st.checkbox("Show Labels on Hover", value=False)
            node_size = st.slider("Node Size", 1, 100, 10)
            edge_thickness = st.slider("Edge Thickness", 0.1, 5.0, 0.5)
            node_shape = st.selectbox(
                "Node Shape",
                [
                    "circle",
                    "square",
                    "diamond",
                    "cross",
                    "x",
                    "triangle-up",
                    "triangle-down",
                    "pentagon",
                    "hexagon",
                    "octagon",
                    "star",
                    "hexagram",
                ],
                index=0,
            )
            edge_color = st.color_picker("Edge Color", "#888")
            apply_changes = st.form_submit_button("Apply Changes")

    tab1, tab2, tab3 = st.tabs(["Upload a File", "Enter Text Manually", "Website URL"])
    user_input = None

    with tab1:
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf"])
        generate_file_input = st.button("Generate Graph from File")

    with tab2:
        text_input = st.text_area(
            "Text Input", height=150, placeholder="Paste your text here..."
        )
        generate_text_input = st.button("Generate Graph from Text")
    
    with tab3:
        url_input = st.text_input("Website URL", placeholder="Enter the website URL here...")
        generate_url_input = st.button("Generate Graph from Website")
    
    user_input = None

    if generate_file_input and uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "text/plain":
            user_input = str(uploaded_file.read(), "utf-8")
        elif file_type == "application/pdf":
            user_input = extract_text_from_pdf(uploaded_file)
        elif (
            file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            user_input = extract_text_from_docx(uploaded_file)
    elif generate_text_input:
        user_input = text_input
    
    if generate_url_input and url_input:
        user_input = extract_text_from_website(url_input)
    
    if user_input:
        st.session_state.user_input = user_input

    if apply_changes or user_input:
        user_input = st.session_state.get("user_input", None)
        if user_input:
            with st.spinner("Generating Knowledge Graph..."):
                try:

                    G, processing_time = build_knowledge_graph(user_input)
                    fig = plot_graph(
                        G,
                        layout_spacing,
                        color_scheme,
                        show_labels,
                        node_size,
                        edge_thickness,
                        show_labels_on_hover,
                        edge_color,
                        node_shape,
                    )

                    plotly_chart_config = {
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": [
                            "zoomIn2d",
                            "zoomOut2d",
                            "autoScale2d",
                            "resetScale2d",
                        ],
                        "modeBarPosition": "bottom",
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "knowledge_graph",
                            "height": 1080,
                            "width": 2048,
                            "scale": 2,
                        },
                    }

                    st.plotly_chart(
                        fig, use_container_width=True, config=plotly_chart_config
                    )

                    (
                        graph_stats,
                        top_nodes,
                        degree_distribution,
                        top_centrality,
                        node_type_counts,
                        num_components,
                        components_color,
                        assortativity_coefficient,
                        top_betweenness_centrality,
                        top_closeness_centrality,
                        top_pagerank,
                    ) = calculate_graph_statistics(G)

                    text_statistics = calculate_text_statistics(user_input)

                    with st.expander("Graph Analytics"):
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(
                            [
                                "Basic Stats",
                                "Degree Stats",
                                "Centrality Measures",
                                "Top Nodes",
                                "Text Stats",
                            ]
                        )

                        with tab1:
                            st.subheader("Basic Graph Statistics")
                            basic_stats_cols = st.columns(2)
                            with basic_stats_cols[0]:
                                safe_metric_display(
                                    "Number of Nodes",
                                    graph_stats.get("Number of Nodes"),
                                )
                                safe_metric_display(
                                    "Number of Edges",
                                    graph_stats.get("Number of Edges"),
                                )
                                safe_metric_display(
                                    "Graph Density", graph_stats.get("Graph Density")
                                )
                                safe_metric_display(
                                    "Connected Components",
                                    graph_stats.get("Connected Components"),
                                )
                                safe_metric_display(
                                    "Average Degree", graph_stats.get("Average Degree")
                                )
                            with basic_stats_cols[1]:
                                safe_metric_display(
                                    "Global Efficiency",
                                    graph_stats.get("Global Efficiency"),
                                )
                                safe_metric_display(
                                    "Local Efficiency",
                                    graph_stats.get("Local Efficiency"),
                                )
                                safe_metric_display(
                                    "Transitivity", graph_stats.get("Transitivity")
                                )
                                safe_metric_display(
                                    "Wiener Index",
                                    graph_stats.get("Wiener Index", "N/A"),
                                )

                        with tab2:
                            st.subheader("Degree Distribution")
                            st.bar_chart(degree_distribution)

                        with tab3:
                            st.subheader("Centrality Measures")
                            centrality_cols = st.columns(2)
                            with centrality_cols[0]:
                                st.table(top_centrality)
                                st.table(top_betweenness_centrality)
                            with centrality_cols[1]:
                                st.table(top_closeness_centrality)
                                st.table(top_pagerank)

                        with tab4:
                            st.subheader("Top Nodes by Connections")
                            st.table(top_nodes)

                        with tab5:
                            st.subheader("Text Statistics")
                            text_stats_cols = st.columns(2)
                            keys = list(text_statistics.keys())
                            half_length = len(keys) // 2
                            for i, key in enumerate(keys):
                                if i < half_length:
                                    with text_stats_cols[0]:
                                        safe_metric_display(key, text_statistics[key])
                                else:
                                    with text_stats_cols[1]:
                                        safe_metric_display(key, text_statistics[key])

                        st.write(f"Processing Time: {processing_time:.2f} seconds")

                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
