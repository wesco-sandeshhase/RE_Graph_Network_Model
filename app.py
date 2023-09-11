import pickle
import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pyvis


# Function to get attributes of nodes connected to the searched node
def get_connected_components(graph, searched_node):
    connected_components = []
    for component in nx.weakly_connected_components(graph):
        if searched_node in component:
            connected_components.append(sorted(component))
    return connected_components


# Function to build the Streamlit app UI
def create_ui(digraph):
    st.title(":green[RecomEngine Cross-Reference Tool]")

    # Custom CSS style for heading and subheadings
    heading_style = """
        <style>
            .title {
                color: #FF4500 !important;
                font-size: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #FF4500;
            }

            h3 { color: #FF8C00 !important; }
        </style>
    """
    st.markdown(heading_style, unsafe_allow_html=True)

    # Sidebar search bar
    searched_node = st.sidebar.text_input("Search Target Product:")

    # Display attributes of searched node
    if searched_node:
        if searched_node in digraph.nodes:
            st.markdown(
                "<h5>Target Product: <span style='color: #7B3F00;'>{}</span></h5>".format(
                    searched_node
                ),
                unsafe_allow_html=True,
            )
            attributes = digraph.nodes[searched_node]
            attributes_df = pd.DataFrame(attributes, index=["Target"]).reset_index(
                drop=True
            )
            attributes_df = attributes_df.style.set_properties(
                subset=pd.IndexSlice[:, :], **{"color": "#7B3F00"}
            )
            # attributes_df = attributes_df.style.set_properties(
            #     subset=["Attribute"], **{"color": "#080b6c", "font-weight": "bold"}
            # )
            # attributes_df = attributes_df.set_properties(
            #     subset=["Value"], **{"color": "#0276ab"}
            # )

            st.dataframe(attributes_df)

            ## Visualize the graph
            # display_connected_nodes(digraph, searched_node, connected_components)

            # Display connected components and their attributes
            st.markdown("<h5>Substitute products:</h5>", unsafe_allow_html=True)
            connected_components = get_connected_components(digraph, searched_node)
            connected_components = [
                [i for i in connected_components[0] if not i == searched_node]
            ]
            print(connected_components, searched_node)
            for i, component in enumerate(connected_components):
                component_attributes = [digraph.nodes[node] for node in component]
                attributes_df = pd.DataFrame(component_attributes)

                attributes_df.insert(0, "sub_id", component)
                col_to_keep = [
                    "sub_id",
                    "mfr_part_num",
                    "mfr_name",
                    "preferred_supplier",
                    "product_type",
                    "wesco_parent",
                    "product_name",
                    "product_description",
                ]
                attributes_df = attributes_df[col_to_keep]
                # attributes_df["sub_id"] = component
                attributes_df = attributes_df.style.set_properties(
                    subset=pd.IndexSlice[:, :], **{"color": "#0276ab"}
                )
                st.dataframe(attributes_df)

        else:
            st.write("Node not found.")
    else:
        st.write("Enter a product in the search field.")

    return searched_node


def display_connected_nodes(graph, node, connected_nodes):
    # connected_nodes = list(graph.successors(node))

    if not connected_nodes:
        st.write(f"No connected nodes found for node '{node}'.")
    else:
        plt.figure(figsize=(10, 6))
        subgraph = graph.subgraph([node] + connected_nodes)
        pos = nx.spring_layout(subgraph)
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_color="#03a9f4",
            node_size=500,
            edge_color="black",
        )
        plt.title(f"Connected Nodes for '{node}'")
        st.pyplot(plt)


# # Streamlit UI code
# st.title("Node Search and Visualization")


def get_connected_components_1(digraph, node):
    # Get the connected components consisting of the specified node and all indirectly connected nodes
    ancestors = set()
    descendants = {node}

    while True:
        prev_len = len(ancestors) + len(descendants)

        new_ancestors = set()
        for n in descendants:
            new_ancestors.update(nx.ancestors(digraph, n))
        ancestors.update(new_ancestors)

        new_descendants = set()
        for n in ancestors:
            new_descendants.update(nx.descendants(digraph, n))
        descendants.update(new_descendants)

        if prev_len == len(ancestors) + len(descendants):
            break

    connected_components = list(ancestors | descendants)
    return digraph.subgraph(connected_components)


def visualize_subgraph(subgraph):
    # Create and configure the pyvis network object
    network = pyvis.network.Network(height="500px", width="1000px", notebook=False)

    # Add nodes and edges to the network from the subgraph
    for n, attrs in subgraph.nodes(data=True):
        network.add_node(n)
    for n1, n2, attrs in subgraph.edges(data=True):
        network.add_edge(n1, n2)

    # Visualize the network
    network.show("subgraph.html")


@st.cache_resource
def load_directed_graph():
    # Generate or load your directed graph here
    digraph = pickle.load(open("graph_xref_model1.pickle", "rb"))
    # Example graph

    return digraph


# Main function
def main():
    # Create a directed graph using NetworkX
    digraph = load_directed_graph()
    # digraph = pickle.load(open("graph_xref_model1.pickle", "rb"))
    # Run the app UI
    searched_node = create_ui(digraph)
    if searched_node:
        subgraph = get_connected_components_1(digraph, searched_node)

        # Visualize the subgraph
        visualize_subgraph(subgraph)

        # Display the subgraph visualization
        st.title("Connected Nodes")
        st.components.v1.html(open("subgraph.html").read(), height=600)


# Run the app
if __name__ == "__main__":
    main()
