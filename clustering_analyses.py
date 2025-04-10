from collections import Counter
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import os
import kaleido
import networkx as nx
import leidenalg as la
import igraph as ig
from sklearn.preprocessing import StandardScaler 

config = {
    'file_path': 'mesh_terms_matrix_5yrs_and_keywords.xlsx',
    'pca_components_to_try': range(1, 7),
    'final_pca_components': 5,
    'dbscan_eps': 0.05,
    'dbscan_min_samples': 2,
    'leiden_resolution_pca': 0.2,
    'leiden_resolution_umap': 0.8,
    'kmeans_n_clusters': 5,
    'silhouette_k_range': range(2, 20),
    'anova_alpha': 0.05,
    'top_n_features_to_plot': 10,
    'cluster_output_path': 'Professors_in_clusters.csv',
    'anova_output_path': 'Significant_terms_per_cluster.csv', 
    'top_mesh_terms_output_path': 'Top_Mesh_Terms_Per_Professor.csv',
}

def load_and_preprocess_data(file_path, index_col='Faculty_Full_Name'):
    """Loads and preprocesses the raw data."""
    raw_data = pd.read_excel(file_path, index_col=index_col)
    raw_data.reset_index(inplace=True)
    faculty_names_df = pd.read_excel(file_path, usecols=['Faculty_Full_Name'])
    feature_matrix = raw_data.drop(columns=['Faculty_Full_Name'])
    raw_data.columns = raw_data.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')
    feature_matrix.columns = feature_matrix.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')
    return raw_data, feature_matrix, faculty_names_df

def fig_show(fig):
    fig.update_layout(plot_bgcolor='#255799')
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

raw_data, feature_matrix, faculty_names_df = load_and_preprocess_data(config['file_path'])

mesh_term_columns = [col for col in feature_matrix.columns]

# Calculate the top 3 MeSH terms for each professor
top_mesh_terms_list = []
faculty_names = []
for index, row in raw_data.iterrows():
    professor_name = row['Faculty_Full_Name']
    mesh_term_counts = Counter()
    for term in mesh_term_columns:
        count = row[term]
        if count > 0:
            mesh_term_counts[term] = count
    top_3_terms = [term for term, count in mesh_term_counts.most_common(3)]
    top_mesh_terms_list.append([top_3_terms])  # Store as a list within a list
    faculty_names.append(professor_name)

top_mesh_terms_df = pd.DataFrame({'Faculty_Full_Name': faculty_names, 'Top_Mesh_Terms': top_mesh_terms_list})
top_mesh_terms_df.set_index('Faculty_Full_Name', inplace=True)
top_mesh_terms_df.to_csv(config['top_mesh_terms_output_path'])

# Set up and run PCA on the raw data (columns = MeSH terms, rows = faculty members, values = frquency of terms)
pca = PCA()
pca_embeddings = pca.fit_transform(feature_matrix)

# Visualize explained variance
explained_variance = pca.explained_variance_ratio_
plt.figure()
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Explained Variance')
plt.show()

# Run PCA scatter plot
## This plot shows that the faculty members are not well separated with just two components. Therefore, additional components are needed to further separate the faculty members. A couple of outliers appear and who may be driving differences.
pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[f'PC{i+1}' for i in range(pca_embeddings.shape[1])])
pca_embeddings_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
pca_embeddings_df = pca_embeddings_df.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(pca_embeddings_df, x='PC1', y='PC2', hover_name='Faculty_Full_Name',
                 hover_data={'PC1': False, 'PC2': False,
                             'Top_Mesh_Terms': True})
fig.show()

# Set up UMAP
## UMAP is a non-linear dimensionality reduction technique that has more power than a PCA.
umap_embeddings = UMAP().fit_transform(feature_matrix) # UMAP is running on the raw data

# Run 2D UMAP plot
# Here, we can manually subset faculty and then rerun the UMAP. This gives us more power to "zoom in". Later, can make this process more automated.
umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=["V1", "V2"])
umap_embeddings_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_embeddings_df = umap_embeddings_df.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_embeddings_df, x="V1", y="V2", title="UMAP", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False,
                             'Top_Mesh_Terms': True},
                 width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_show(fig)

# Run an elbow plot
## An elbow plot gives a rough value for the number of components to run UMAP on. Then, we will plug this number (where the elbow bends) into the num_components parameter in UMAP.
numeric_data_umap = umap_embeddings_df.select_dtypes(include=['number'])
scaler = StandardScaler()
scaled_data_umap = scaler.fit_transform(numeric_data_umap)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
    kmeans.fit(scaled_data_umap)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (UMAP)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Run UMAP with PCA components
# UMAP takes into account different dimensions and represnts the information in 2D. If you want smaller and more refined clusters, then use more components. But the starting number of components is usually based on the elbow plot.
pca_result = PCA().fit_transform(feature_matrix)
num_components = 3
pca_reduced_features = pca_result[:, :num_components]
umap_result = UMAP().fit_transform(pca_reduced_features)
umap_df_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_df_pca = umap_df_pca.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_df_pca, x="V1", y="V2", title="UMAP on PCA Components", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False,
                             'Top_Mesh_Terms': True},
                 width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_show(fig)

# Run UMAPs by iterating through different number of PCA components
for num_components in config['pca_components_to_try']:
    pca_reduced_features = pca_result[:, :num_components]
    umap_result = UMAP(random_state=123).fit_transform(pca_reduced_features)
    umap_df_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
    umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
    umap_df_pca = umap_df_pca.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')
    fig = px.scatter(umap_df_pca, x="V1", y="V2", title=f"UMAP with {num_components} PCA Components",
                     hover_name="Faculty_Full_Name",
                     hover_data={"V1": False, "V2": False,
                                 'Top_Mesh_Terms': True},
                     width=800, height=800, color_discrete_sequence=['#fecc07'])
    fig_show(fig)

# Update the number of components after iteration and looking at the elbow plot. This update will be used for the rest of the analysis.
num_components = 5
pca_reduced_features = pca_result[:, :num_components]
umap_result = UMAP(random_state=123).fit_transform(pca_reduced_features)
umap_df_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_df_pca = umap_df_pca.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')

# Cluster UMAP with DBSCAN
knn = NearestNeighbors(n_neighbors=2)
knn.fit(pca_reduced_features)
distances, indices = knn.kneighbors(pca_reduced_features)
dbscan_model = DBSCAN(eps=0.05, min_samples=2).fit(pca_reduced_features)
umap_df_pca['cluster'] = dbscan_model.labels_
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'] # Add faculty names back to dataframe

# This section runs two different clustering methods: Leiden and K-means. The first one is on the PCA scores, and the second one is on the UMAP coordinates. However, the Leiden clustering is not working as well as K-means right now. K-mean is fine to use for the final analysis.

### Leiden clustering on PCA scores
# Create a graph from the KNN results
graph = nx.Graph()
for i in range(len(pca_reduced_features)):
    for j in indices[i]:
        if i != j:  # Avoid self-loops
            graph.add_edge(i, j)

# Convert NetworkX graph to igraph for Leiden algorithm
g_ig = ig.Graph.from_networkx(graph)

# Set resolution parameter (higher values = more clusters)
resolution_parameter = 0.2  # You can adjust this value to get desired number of clusters**

# Run Leiden clustering with resolution parameter
partition = la.find_partition(
    g_ig, 
    la.CPMVertexPartition, 
    resolution_parameter=resolution_parameter
)

# Add Leiden cluster assignments to the dataframe
umap_df_pca['leiden_cluster'] = [membership for membership in partition.membership]

# Print the number of clusters found
num_clusters = len(set(partition.membership))
# print(f"Number of clusters found: {num_clusters}")

# Visualize UMAP with Leiden clusters
fig = px.scatter(
    umap_df_pca,
    x="V1",
    y="V2",
    title="UMAP with Leiden Clustering",
    color="leiden_cluster",
    color_discrete_sequence=px.colors.qualitative.Bold,  # Use a discrete color palette
    category_orders={"leiden_cluster": sorted(umap_df_pca["leiden_cluster"].unique())},  # Order the categories
    hover_name="Faculty_Full_Name",
    hover_data={"V1": False, "V2": False,
                'Top_Mesh_Terms': True},
    width=800,
    height=800
)
fig_show(fig)

### Leiden clustering on UMAP coordinates
# Create a k-nearest neighbors graph from UMAP coordinates instead of PCA scores
k = 100  # You can adjust this parameter
knn = NearestNeighbors(n_neighbors=k)
knn.fit(umap_result)  # Using UMAP coordinates instead of PCA scores
distances, indices = knn.kneighbors(umap_result)

# Create a graph from the KNN results
graph = nx.Graph()
for i in range(len(umap_result)):
    for j in indices[i]:
        if i != j:  # Avoid self-loops
            graph.add_edge(i, j)

# Convert NetworkX graph to igraph for Leiden algorithm
g_ig = ig.Graph.from_networkx(graph)

# Set resolution parameter (higher values = more clusters)
resolution_parameter = 0.8  # You can adjust this value to get desired number of clusters

# Run Leiden clustering with resolution parameter
partition = la.find_partition(
    g_ig, 
    la.CPMVertexPartition, 
    resolution_parameter=resolution_parameter
)

# Add Leiden cluster assignments to the dataframe
umap_df_pca['leiden_cluster'] = [membership for membership in partition.membership]

# Print the number of clusters found
num_clusters = len(set(partition.membership))
# print(f"Number of clusters found: {num_clusters}")

# Visualize UMAP with Leiden clusters
fig = px.scatter(
    umap_df_pca,
    x="V1",
    y="V2",
    title=f"UMAP with Leiden Clustering (resolution={resolution_parameter}, clusters={num_clusters})",
    color="leiden_cluster",
    color_discrete_sequence=px.colors.qualitative.Bold,  # Use a discrete color palette
    category_orders={"leiden_cluster": sorted(umap_df_pca["leiden_cluster"].unique())},  # Order the categories
    hover_name="Faculty_Full_Name",
    hover_data={"V1": False, "V2": False,
                'Top_Mesh_Terms': True},
    width=800,
    height=800
)
fig_show(fig)

# Identify optimal number of clusters using silhouette score
## Can go with highest peak but then won't include as many clusters. Can have a rule that I want X number of clusters.
def avg_silhouette(data, k):
    kmeans = KMeans(n_clusters=k, random_state=123, n_init="auto").fit(data)
    labels = kmeans.labels_
    sil_score = silhouette_score(data, labels)
    return sil_score

# Calculate silhouette scores for different K
# Min number of clusters is 2. Max number of clusters is 50.
k_values = range(2, 50)
# umap_df_pca = umap_df_pca.drop(columns=['Faculty_Full_Name'])
# silhouette_scores = [avg_silhouette(umap_df_pca, k) for k in k_values]
# plt.figure()
# plt.plot(k_values, silhouette_scores, 'b*-')
# plt.xlabel('Number of clusters K')
# plt.ylabel('Average Silhouette Width')
# plt.title('Silhouette Score for Different K')
# plt.show()
umap_embeddings_for_silhouette = umap_df_pca[['V1', 'V2']].copy()

silhouette_scores = [avg_silhouette(umap_embeddings_for_silhouette, k) for k in k_values]
plt.figure()
plt.plot(k_values, silhouette_scores, 'b*-')
plt.xlabel('Number of clusters K')
plt.ylabel('Average Silhouette Width')
plt.title('Silhouette Score for Different K')
plt.show()

### K-means clustering on UMAP coordinates
# Number of clusters for K-means
n_clusters = 8 # You can adjust this parameter to get desired number of clusters

# Perform K-means clustering on UMAP coordinates
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(umap_result)

# Add K-means cluster assignments to the dataframe
umap_df_pca['kmeans_cluster'] = cluster_labels

# Print the number of clusters
print(f"Number of K-means clusters: {n_clusters}")

# Visualize UMAP with K-means clusters
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
fig = px.scatter(
    umap_df_pca,
    x="V1",
    y="V2",
    color='kmeans_cluster',
    title=f"UMAP with K-means Clusters (k={n_clusters})",
    hover_name="Faculty_Full_Name",
    hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
    width=800,
    height=800,
    color_discrete_sequence=px.colors.qualitative.Bold  # Use a discrete color palette
)
fig_show(fig)

# Process clustering results and prepare for analysis
## This block creates a dataframe with numeric data, adds cluster labels, ensures proper data types, and identifies feature columns for further analysis
filtered_data_df = pd.DataFrame(feature_matrix)  # Create a new dataframe with numeric data
filtered_data_df['cluster'] = kmeans.labels_  # Add cluster labels to the dataframe

# Create a clustered dataframe for analysis
cluster_feature_matrix = filtered_data_df

# Ensure the cluster column is properly formatted
cluster_feature_matrix['cluster'] = cluster_feature_matrix['cluster'].astype('category')

# Get feature names (all columns except 'cluster')
feature_names = [col for col in cluster_feature_matrix.columns if col != 'cluster']

# Function to run ANOVA for each feature
def get_anova_pvalues(feature, data):
    model = ols(f"{feature} ~ C(cluster)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table["PR(>F)"][0]  # p-value

# Calculate p-values for all features
feature_p_values = {}
for feature in feature_names:
    try:
        feature_p_values[feature] = get_anova_pvalues(feature, cluster_feature_matrix)
    except:
        print(f"Error processing feature: {feature}")
        feature_p_values[feature] = 1.0  # Default to not significant

# Adjust p-values using Benjamini-Hochberg method (FDR correction)
feature_list = list(feature_p_values.keys())
pval_list = list(feature_p_values.values())
_, adjusted_p_values, _, _ = multipletests(pval_list, method='fdr_bh', alpha=0.05)

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': feature_list,
    'p_value': pval_list,
    'adjusted_p_values': adjusted_p_values,
    'significant': adjusted_p_values < 0.05
})

# Sort by adjusted p-value
results_df = results_df.sort_values('adjusted_p_values')

# Print significant features
significant_features_df = results_df[results_df['significant']]
print(f"Found {len(significant_features_df)} significant features:")
print(significant_features_df[['Feature', 'adjusted_p_values']])

# Visualize top significant features
def plot_top_features(cluster_feature_matrix, significant_features_df, top_n=10):
    top_features = significant_features_df['Feature'].head(top_n).tolist()
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 5, i+1)
        sns.boxplot(x='cluster', y=feature, data=cluster_feature_matrix)
        plt.title(f"{feature.split('Normalized_')[-1][:15]}...")
        plt.tight_layout()
    
    plt.savefig('top_significant_features.png')
    plt.show()

# If there are significant features, plot them
if len(significant_features_df) > 0:
    plot_top_features(cluster_feature_matrix, significant_features_df)

# ADD NEW CODE BLOCK #3 HERE - Calculate Within-Cluster Similarity
# Calculate average similarity within each cluster
within_cluster_similarity = {}
for cluster_id in cluster_feature_matrix['cluster'].unique():
    # Get data for this cluster
    cluster_data = cluster_feature_matrix[cluster_feature_matrix['cluster'] == cluster_id][feature_names]
    
    # Skip if only one member in cluster
    if len(cluster_data) <= 1:
        within_cluster_similarity[cluster_id] = float('nan')
        continue
        
    # Calculate pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(cluster_data)
    
    # Average similarity (excluding self-similarity on diagonal)
    n = similarities.shape[0]
    total_sim = similarities.sum() - n  # Subtract diagonal (self-similarities)
    avg_sim = total_sim / (n * (n - 1))  # n(n-1) pairs
    
    within_cluster_similarity[cluster_id] = avg_sim

# Create DataFrame and save
similarity_df = pd.DataFrame.from_dict(within_cluster_similarity, 
                                      orient='index', 
                                      columns=['Average_Similarity'])
similarity_df.to_csv("Within_cluster_similarity.csv")

# ADD NEW CODE BLOCK #4 HERE - Visualize Cluster Profiles
# Create a heatmap of top terms across clusters
top_n = 20  # Number of top terms to include (adjust as needed)
significant_terms = significant_features_df['Feature'].head(top_n).tolist()

# Calculate cluster means for these terms
cluster_profiles = cluster_feature_matrix.groupby('cluster')[significant_terms].mean()

# Plot heatmap
plt.figure(figsize=(10, 14))  # Tall figure
# Transpose the dataframe so terms are rows
sns.heatmap(cluster_profiles.T, cmap="YlGnBu", annot=False)
plt.title("Cluster Profiles: Top Significant Terms")
plt.tight_layout()
plt.savefig("Cluster_profiles_transposed.png", dpi=300, bbox_inches='tight')
plt.show()

filtered_data_df.columns = filtered_data_df.columns.str.replace(' ', '_').str.replace(',', '_').str.replace('-', '_')
feature_names = filtered_data_df.columns[:-1]
umap_df_pca = umap_df_pca.drop(columns=['Faculty_Full_Name'])

# Display all unique mesh terms associated with professor
def get_faculty_mesh_terms(faculty_list, raw_data, mesh_term_columns):
    faculty_mesh_terms = {}
    all_terms = []

    for faculty_name in faculty_list:
        if faculty_name in raw_data['Faculty_Full_Name'].values:
            faculty_row = raw_data[raw_data['Faculty_Full_Name'] == faculty_name].iloc[0]
            unique_terms = [term for term in mesh_term_columns if faculty_row[term] > 0]
            faculty_mesh_terms[faculty_name] = unique_terms
            all_terms.extend(unique_terms)
        else:
            faculty_mesh_terms[faculty_name] = f"Faculty member '{faculty_name}' not found in the data."

    # Find overlapping terms
    term_counts = Counter(all_terms)
    overlapping_terms = {term for term, count in term_counts.items() if count == len(faculty_list) and term in [term for terms in faculty_mesh_terms.values() if isinstance(terms, list) for term in terms]}

    return faculty_mesh_terms, overlapping_terms

# Example Usage (assuming your raw_data and mesh_term_columns are already defined):
faculty_to_check = ["Bracken, Matthew", "Allison, Steven", "Treseder, Kathleen"]
faculty_mesh_results, common_terms = get_faculty_mesh_terms(faculty_to_check, raw_data, mesh_term_columns)

for faculty, terms in faculty_mesh_results.items():
    print(f"{faculty}: {terms}")

print("\nOverlapping MeSH terms:")
print(common_terms)

# Save outputs
sig_df_path = "Significant_terms_per_cluster.csv"
cluster_df_path = "Professors_in_clusters.csv"
fig_path = "UMAP_professors_clusters.pdf"

# Your data manipulation and saving logic (unchanged)
umap_df_pca = umap_df_pca.join(faculty_names_df, how='left')
umap_df_pca = umap_df_pca.groupby('cluster')['Faculty_Full_Name'].apply(list).reset_index()
umap_df_pca.to_csv(cluster_df_path, index=True)

# print("significant_features_df:", significant_features_df['Feature'].tolist()) #added tolist()
# print("adjusted_p_values:", results_df['adjusted_p_values'].tolist()) #added print statement to see adjusted_p_values values.

significant_features_output_df = pd.DataFrame({
    'Feature': significant_features_df['Feature'].tolist(),  # Extract the 'Feature' column as a list
    'P-Value Adjusted': results_df['adjusted_p_values'].tolist()[:len(significant_features_df)] #use results_df, and slice to the correct length.
})
significant_features_output_df.to_csv(sig_df_path, index=True)