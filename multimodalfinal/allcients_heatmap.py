import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for each client
client_files = {
    'Client 1': '/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_image_final_all_patients_embeddings.csv',
    'Client 2': '/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client2/c2_image_final_all_patients_embeddings.csv',
    'Client 3': '/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client3/c3_image_final_all_patients_embeddings.csv'
}

# Initialize a dataframe to store client-wise average activations
client_activations = {}

# Load embeddings and calculate mean activation for each dimension per client
for client, file_path in client_files.items():
    df = pd.read_csv(file_path)
    
    # Drop patient_id and label columns, keep only embeddings
    embedding_columns = [col for col in df.columns if col.startswith("dim_")]
    embeddings = df[embedding_columns].values
    
    # Compute mean activation per dimension
    mean_activations = np.mean(embeddings, axis=0)
    
    # Store in dictionary
    client_activations[client] = mean_activations

# Convert dictionary to DataFrame
activations_df = pd.DataFrame.from_dict(client_activations, orient='index', columns=embedding_columns)

# Select top 15 dimensions with the lowest mean activations
top_15_dims = activations_df.mean(axis=0).nsmallest(15).index.tolist()
top_activations_df = activations_df[top_15_dims]

# Plot heatmap
plt.figure(figsize=(14, 5))
sns.heatmap(
    top_activations_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Mean Activation'}
)
plt.title("Client-wise Average Image Feature Activations (Top 15 Dimensions)", fontsize=14)
plt.xlabel("Image Embedding Dimensions", fontsize=12)
plt.ylabel("Client", fontsize=12)
plt.tight_layout()
plt.show()
