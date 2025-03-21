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

# Import raw data matrix
rawData = pd.read_excel('mesh_terms_matrix_5yrs_and_keywords.xlsx', index_col = 'Faculty_Full_Name')
rawData.reset_index(inplace=True) 
numeric_data = rawData.drop(columns=['Faculty_Full_Name'])
faculty_column = pd.read_excel('mesh_terms_matrix_5yrs_and_keywords.xlsx', usecols=['Faculty_Full_Name'])
rawData.columns = rawData.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_') # Remove spaces from column names

# Set up and run PCA
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

# PCA scatter plot
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
fig = px.scatter(pca_df, x='PC1', y='PC2')
fig.show()

# Set up and run UMAP
umap_data = UMAP().fit_transform(numeric_data)

# 2D UMAP plot
umap_df = pd.DataFrame(umap_data, columns=["V1", "V2"])
umap_df['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(umap_df, x="V1", y="V2", title="UMAP", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Run t-SNE and show results
tsne = TSNE(n_components=2, perplexity=25)
tsne_result = tsne.fit_transform(numeric_data)
tsne_df = pd.DataFrame(tsne_result, columns=["V1", "V2"])
tsne_df['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(tsne_df, x="V1", y="V2", title="t-SNE", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Run UMAP with PCA components
pca_result = PCA().fit_transform(numeric_data)
num_components = 2
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

# Run UMAP with different number of PCA components
for num_components in range(1, 20):
    pca_scores = pca_result[:, :num_components]
    umap_result = UMAP().fit_transform(pca_scores)
    umapDf_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
    umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
    fig = px.scatter(umapDf_pca, x="V1", y="V2", title=f"UMAP with {num_components} PCA Components", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
    fig.update_layout(plot_bgcolor='#255799')
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

# Cluster UMAP with DBSCAN
knn = NearestNeighbors(n_neighbors=8)
knn.fit(pca_scores)
distances, indices = knn.kneighbors(pca_scores)
dbs = DBSCAN(eps=0.05, min_samples=2).fit(pca_scores)
umapDf_pca['cluster'] = dbs.labels_

# Plot UMAP with clusters
fig = px.scatter(umapDf_pca, x="V1", y="V2", color='cluster', title="UMAP with Clusters", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name'] # Add faculty names back to dataframe

# K-means clustering
kmeans = KMeans(n_clusters=12, random_state=123).fit(pca_scores)
umapDf_pca['cluster'] = kmeans.labels_

# Plot UMAP with K-means clusters
fig = px.scatter(umapDf_pca, x="V1", y="V2", color='cluster', title="UMAP with K-means Clusters", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

# Identify optimal number of clusters using silhouette score
def avg_silhouette(data, k):
    kmeans = KMeans(n_clusters=k, random_state=123).fit(data)
    labels = kmeans.labels_
    sil_score = silhouette_score(data, labels)
    return sil_score

# Calculate silhouette scores for different K
k_values = range(2, 21)
umapDf_pca = umapDf_pca.drop(columns=['Faculty_Full_Name'])
sil_values = [avg_silhouette(umapDf_pca, k) for k in k_values]
plt.figure()
plt.plot(k_values, sil_values, 'b*-')
plt.xlabel('Number of clusters K')
plt.ylabel('Average Silhouette Width')
plt.title('Silhouette Score for Different K')
plt.show()

filtered_data_df = pd.DataFrame(numeric_data) # Create a new dataframe with numeric data
filtered_data_df['cluster'] = kmeans.labels_ # Add cluster labels to the dataframe

# Perform ANOVA and identify significant features
def get_anova_pvalues(feature, data):
    model = ols(f"{feature} ~ C(cluster)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table["PR(>F)"][0]

filtered_data_df.columns = filtered_data_df.columns.str.replace(' ', '_').str.replace(',', '_').str.replace('-', '_')
feature_names = filtered_data_df.columns[:-1]
# print(feature_names), print(filtered_data_df.columns)
# print(filtered_data_df.dtypes)
pvalues = {feature: get_anova_pvalues(feature, filtered_data_df) for feature in feature_names}

# Adjust p-values using Benjamini-Hochberg method
_, p_adjusted, _, _ = multipletests(list(pvalues.values()), method='fdr_bh')
significant_features = [feature for feature, pval in zip(feature_names, p_adjusted) if pval < 0.05]
print("Significant features:", significant_features)

# Plot UMAP result with clusters
umapDf_pca['Faculty_Full_Name'] = rawData['Faculty_Full_Name']
fig = px.scatter(umapDf_pca, x="V1", y="V2", color='cluster', title="UMAP with K-means Clusters", hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig.update_layout(plot_bgcolor='#255799')
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
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
    'Feature': significant_features,
    'P-Value Adjusted': p_adjusted[:len(significant_features)]
})
sig_df.to_csv(sig_df_path, index=True)