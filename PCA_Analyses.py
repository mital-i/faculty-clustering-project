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

# Import raw data matrix
rawData = pd.read_excel('mesh_terms_matrix_5yrs_and_keywords.xlsx', index_col = 'Faculty_Full_Name')
rawData.reset_index(inplace=True) 
numeric_data = rawData.drop(columns=['Faculty_Full_Name'])
faculty_column = pd.read_excel('mesh_terms_matrix_5yrs_and_keywords.xlsx', usecols=['Faculty_Full_Name'])
rawData.columns = rawData.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_') # Remove spaces from column names

# Set up and run PCA on the raw data (columns = MeSH terms, rows = faculty members, values = frquency of terms)
pca = PCA()
pca_data = pca.fit_transform(numeric_data)

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
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
fig = px.scatter(pca_df, x='PC1', y='PC2')
fig.show()

# Set up UMAP
## UMAP is a non-linear dimensionality reduction technique that has more power than a PCA.
umap_data = UMAP().fit_transform(numeric_data) # UMAP is running on the raw data ("numeric_data")

# Run 2D UMAP plot
# Here, we can manually subset faculty and then rerun the UMAP. This gives us more power to "zoom in". Later, can make this process more automated.
umap_df = pd.DataFrame(umap_data, columns=["V1", "V2"])
umap_df['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(umap_df, x="V1", y="V2", title="UMAP", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Run an elbow plot
## We will need to code this. An elbow plot gives a rough value for the number of components to run UMAP on. Then, we will plug this number (where the elbow bends) into the num_components parameter in UMAP.

# Run UMAP with PCA components
# UMAP takes into account different dimensions and represnts the information in 2D. If you want smaller and more refined clusters, then use more components. But the starting number of components is usually based on the elbow plot.
pca_result = PCA().fit_transform(numeric_data)
num_components = 3
pca_scores = pca_result[:, :num_components]
umap_result = UMAP().fit_transform(pca_scores)
umapDf_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(umapDf_pca, x="V1", y="V2", title="UMAP on PCA Components", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Run UMAPs by iterating through different number of PCA components
for num_components in range(1, 7):
    pca_scores = pca_result[:, :num_components]
    umap_result = UMAP(random_state=123).fit_transform(pca_scores)
    umapDf_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
    umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
    fig = px.scatter(umapDf_pca, x="V1", y="V2", title=f"UMAP with {num_components} PCA Components", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
    fig.update_layout(plot_bgcolor='#255799')
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

# Update the number of components after iteration and looking at the elbow plot. This update will be used for the rest of the analysis.
num_components = 5
pca_scores = pca_result[:, :num_components]
umap_result = UMAP(random_state=123).fit_transform(pca_scores)
umapDf_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']

# Cluster UMAP with DBSCAN
knn = NearestNeighbors(n_neighbors=2)
knn.fit(pca_scores)
distances, indices = knn.kneighbors(pca_scores)
dbs = DBSCAN(eps=0.05, min_samples=2).fit(pca_scores)
umapDf_pca['cluster'] = dbs.labels_
umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name'] # Add faculty names back to dataframe

# This section runs two different clustering methods: Leiden and K-means. The first one is on the PCA scores, and the second one is on the UMAP coordinates. However, the Leiden clustering is not working as well as K-means right now. K-mean is fine to use for the final analysis.

### Leiden clustering on PCA scores
# Create a graph from the KNN results
graph = nx.Graph()
for i in range(len(pca_scores)):
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
umapDf_pca['leiden_cluster'] = [membership for membership in partition.membership]

# Print the number of clusters found
num_clusters = len(set(partition.membership))
print(f"Number of clusters found: {num_clusters}")

# Visualize UMAP with Leiden clusters
fig = px.scatter(
    umapDf_pca, 
    x="V1", 
    y="V2", 
    title="UMAP with Leiden Clustering", 
    # hover_name="Faculty_Full_Name", 
    color="leiden_cluster",
    color_discrete_sequence=px.colors.qualitative.Bold,  # Use a discrete color palette
    category_orders={"leiden_cluster": sorted(umapDf_pca["leiden_cluster"].unique())},  # Order the categories
    hover_data={"V1": False, "V2": False}, 
    width=800, 
    height=800
)
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="", showticklabels=False)
fig.update_yaxes(title_text="", showticklabels=False)
fig.show()

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
umapDf_pca['leiden_cluster'] = [membership for membership in partition.membership]

# Print the number of clusters found
num_clusters = len(set(partition.membership))
print(f"Number of clusters found: {num_clusters}")

# Visualize UMAP with Leiden clusters
fig = px.scatter(
    umapDf_pca, 
    x="V1", 
    y="V2", 
    title=f"UMAP with Leiden Clustering (resolution={resolution_parameter}, clusters={num_clusters})", 
    color="leiden_cluster",
    color_discrete_sequence=px.colors.qualitative.Bold,  # Use a discrete color palette
    category_orders={"leiden_cluster": sorted(umapDf_pca["leiden_cluster"].unique())},  # Order the categories
    hover_data={"V1": False, "V2": False}, 
    width=800, 
    height=800
)
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="", showticklabels=False)
fig.update_yaxes(title_text="", showticklabels=False)
fig.show()

# Identify optimal number of clusters using silhouette score
## Can go with highest peak but then won't include as many clusters. Can have a rule that I want X number of clusters.
def avg_silhouette(data, k):
    kmeans = KMeans(n_clusters=k, random_state=123).fit(data)
    labels = kmeans.labels_
    sil_score = silhouette_score(data, labels)
    return sil_score

# Calculate silhouette scores for different K
# Min number of clusters is 2. Max number of clusters is 50.
k_values = range(2, 50)
umapDf_pca = umapDf_pca.drop(columns=['Faculty_Full_Name'])
sil_values = [avg_silhouette(umapDf_pca, k) for k in k_values]
plt.figure()
plt.plot(k_values, sil_values, 'b*-')
plt.xlabel('Number of clusters K')
plt.ylabel('Average Silhouette Width')
plt.title('Silhouette Score for Different K')
plt.show()

### K-means clustering on UMAP coordinates
# Number of clusters for K-means
n_clusters = 5  # You can adjust this parameter to get desired number of clusters

# Perform K-means clustering on UMAP coordinates
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(umap_result)

# Add K-means cluster assignments to the dataframe
umapDf_pca['kmeans_cluster'] = cluster_labels

# Print the number of clusters
print(f"Number of K-means clusters: {n_clusters}")

# Visualize UMAP with K-means clusters
umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(
    umapDf_pca, 
    x="V1", 
    y="V2", 
    color='kmeans_cluster', 
    title=f"UMAP with K-means Clusters (k={n_clusters})", 
    hover_name="Faculty_Full_Name", 
    hover_data={"V1": False, "V2": False}, 
    width=800, 
    height=800, 
    color_discrete_sequence=px.colors.qualitative.Bold  # Use a discrete color palette
)
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Process clustering results and prepare for analysis
## This block creates a dataframe with numeric data, adds cluster labels, ensures proper data types, and identifies feature columns for further analysis
filtered_data_df = pd.DataFrame(numeric_data)  # Create a new dataframe with numeric data
filtered_data_df['cluster'] = kmeans.labels_  # Add cluster labels to the dataframe

# Create a clustered dataframe for analysis
clustered_df = filtered_data_df

# Ensure the cluster column is properly formatted
clustered_df['cluster'] = clustered_df['cluster'].astype('category')

# Get feature names (all columns except 'cluster')
feature_names = [col for col in clustered_df.columns if col != 'cluster']

# Function to run ANOVA for each feature
def get_anova_pvalues(feature, data):
    model = ols(f"{feature} ~ C(cluster)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table["PR(>F)"][0]  # p-value

# Calculate p-values for all features
pvalues = {}
for feature in feature_names:
    try:
        pvalues[feature] = get_anova_pvalues(feature, clustered_df)
    except:
        print(f"Error processing feature: {feature}")
        pvalues[feature] = 1.0  # Default to not significant

# Adjust p-values using Benjamini-Hochberg method (FDR correction)
feature_list = list(pvalues.keys())
pval_list = list(pvalues.values())
_, p_adjusted, _, _ = multipletests(pval_list, method='fdr_bh', alpha=0.05)

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': feature_list,
    'p_value': pval_list,
    'p_adjusted': p_adjusted,
    'significant': p_adjusted < 0.05
})

# Sort by adjusted p-value
results_df = results_df.sort_values('p_adjusted')

# Print significant features
sig_features = results_df[results_df['significant']]
print(f"Found {len(sig_features)} significant features:")
print(sig_features[['Feature', 'p_adjusted']])

# Visualize top significant features
def plot_top_features(clustered_df, sig_features, top_n=16):
    top_features = sig_features['Feature'].head(top_n).tolist()
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 5, i+1)
        sns.boxplot(x='cluster', y=feature, data=clustered_df)
        plt.title(f"{feature.split('Normalized_')[-1][:15]}...")
        plt.tight_layout()
    
    plt.savefig('top_significant_features.png')
    plt.show()

# If there are significant features, plot them
if len(sig_features) > 0:
    plot_top_features(clustered_df, sig_features)

filtered_data_df.columns = filtered_data_df.columns.str.replace(' ', '_').str.replace(',', '_').str.replace('-', '_')
feature_names = filtered_data_df.columns[:-1]
umapDf_pca = umapDf_pca.drop(columns=['Faculty_Full_Name'])

# Save outputs
path = '/Users/justinsarkis/Downloads/'
sig_df_path = os.path.join(path, "Significant_terms_per_cluster.csv")
cluster_df_path = os.path.join(path, "Professors_in_clusters.csv")
fig_path = os.path.join(path, "UMAP_professors_clusters.pdf")

umapDf_pca = umapDf_pca.join(faculty_column, how='left')
umapDf_pca = umapDf_pca.groupby('cluster')['Faculty_Full_Name'].apply(list).reset_index()
umapDf_pca.to_csv(cluster_df_path,index=True)

sig_df = pd.DataFrame({
    'Feature': sig_features,
    'P-Value Adjusted': p_adjusted[:len(sig_features)]
})
sig_df.to_csv(sig_df_path, index=True)