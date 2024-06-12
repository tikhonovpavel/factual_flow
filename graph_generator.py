import torch
from graphviz import Digraph
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

import re

def generate_graph(model, tokenizer, text, depth=4, top_k=2):
    token_ids_by_node = {}  # Dictionary to store lists of token IDs by node identifiers

    dot = Digraph(comment='Prediction Tree', graph_attr={'size': '11,11'})
    dot.attr('node', fontsize='14')
    unique_id = 0

    vocab = tokenizer.get_vocab()
    vocab_reverse = {v: k for k, v in vocab.items()}
    tokenizer_word_sep = tokenizer.tokenize(' hello')[0][0]

    def add_nodes_and_edges(node_id, current_depth=1):
        nonlocal unique_id
        if current_depth > depth:
            return

        # Get the current token IDs for this node_id
        current_token_ids = token_ids_by_node[node_id]
        inputs = torch.tensor([current_token_ids]).to(model.device)

        with torch.no_grad():
            predictions = model(inputs)[0]

        # Apply softmax to the entire model output for the last token
        probabilities = torch.softmax(predictions[0, -1, :], dim=0)

        top_scores, predicted_indices = torch.topk(probabilities, top_k)

        for i, (idx, score) in enumerate(zip(predicted_indices.tolist(), top_scores.tolist())):
            new_token_ids = current_token_ids + [idx]
            child_token = vocab_reverse[idx]
            child_token_clean = clear_text(tokenizer_word_sep, child_token)  # Clean only the last token for the node name

            child_unique_id = f"node{unique_id}"
            unique_id += 1

            # Save the new list of token IDs for the new node
            token_ids_by_node[child_unique_id] = new_token_ids

            edge_label = f"{score:.2f}"

            dot.node(child_unique_id, child_token_clean)
            dot.edge(node_id, child_unique_id, label=edge_label)
            
            add_nodes_and_edges(child_unique_id, current_depth + 1)

    start_node = f"node{unique_id}"
    unique_id += 1
    initial_token_ids = tokenizer.encode(text, add_special_tokens=True)
    token_ids_by_node[start_node] = initial_token_ids  # Save token IDs for the root node
    dot.node(start_node, clear_text(tokenizer_word_sep, text))  # Use cleaned text for the root node
    add_nodes_and_edges(start_node)

    return dot

def generate_and_display_graphs(model, tokenizer, text1, text2, depth=4, top_k=2):
    def digraph_to_image(dot):
        dot.attr(dpi='200')
        png_image = dot.pipe(format='png')
        image = Image.open(BytesIO(png_image))
        return image

    # Generate graphs
    g1 = generate_graph(model, tokenizer, text1, depth=depth, top_k=top_k)
    g2 = generate_graph(model, tokenizer, text2, depth=depth, top_k=top_k)

    # Convert graphs to images
    image1 = digraph_to_image(g1)
    image2 = digraph_to_image(g2)

    # Create subplots and display images
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].imshow(image1)
    axs[0].axis('off')  # Remove axes for the first graph
    axs[1].imshow(image2)
    axs[1].axis('off')  # Remove axes for the second graph

    plt.tight_layout()
    plt.show()
    
    return fig

def clear_text(tokenizer_word_sep, text):
    return re.sub(f'[^\w\- {tokenizer_word_sep}?#$%&*]+', '_', text)
