import pandas as pd
import numpy as np
import streamlit as st
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
from collections import Counter
# Assuming Biopython_Entrez exists or commenting out if not
# from Biopython_Entrez import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import os
import kaleido

# Load raw data
# Basic error check for file loading
try:
    data_path = 'mesh_terms_matrix_5yrs_and_keywords.xlsx'
    raw_data = pd.read_excel(data_path, index_col='Faculty_Full_Name')
    raw_data.reset_index(inplace=True)
except FileNotFoundError:
    st.error(
        f"Error: The file '{data_path}' was not found. Please ensure it's in the correct directory.")
    st.stop()  # Stop execution if file not found

# Extract numeric data and faculty column
numeric_data = raw_data.drop(columns=['Faculty_Full_Name'])
# faculty_column = pd.read_excel(data_path, usecols=['Faculty_Full_Name']) # Not strictly needed if index is used

# Clean column names (using original simple cleaning)
raw_data.columns = raw_data.columns.str.replace(
    ' ', '_').str.replace('-', '_').str.replace(',', '_')
# Apply cleaning to numeric_data columns as well for consistency
numeric_data.columns = numeric_data.columns.str.replace(
    ' ', '_').str.replace('-', '_').str.replace(',', '_')


# PCA
pca = PCA()
# Basic check if numeric_data is actually numeric before fitting PCA
if not all(numeric_data.dtypes.apply(pd.api.types.is_numeric_dtype)):
    st.error("Error: Non-numeric data detected after cleaning. Cannot perform PCA.")
    st.stop()
pca_result = pca.fit_transform(numeric_data)

# Visualize explained variance
st.subheader("PCA Explained Variance")  # Add subheader for clarity
# Write description for Explained Variance Plot using st.write()
st.write("""
**Plot Description:** This plot shows the cumulative percentage of variance captured by the principal components (PCs). The x-axis represents the number of PCs included, and the y-axis shows the total variance explained. It helps determine how many PCs are needed to retain a significant amount of information.
""")
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure()
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
st.pyplot(plt.gcf())  # Original call
plt.clf()  # Add plt.clf() after st.pyplot to prevent figure state issues

# PCA scatter plot
st.subheader("PCA Scatter Plot (PC1 vs PC2)")  # Add subheader
# Write description for PCA Scatter Plot using st.write()
st.write("""
**Plot Description:** This scatter plot displays the data projected onto the first two principal components (PC1 and PC2). Each point represents a faculty member. Reveals primary axes of variation based on PCA reduction.
""")
pca_df = pd.DataFrame(pca_result, columns=[
                      f'PC{i+1}' for i in range(pca_result.shape[1])])
# Use new fig variable name
fig_pca = px.scatter(pca_df, x='PC1', y='PC2',
                     title="PCA Scatter Plot (PC1 vs PC2)")
st.plotly_chart(fig_pca)  # Original call modified to use new variable

# Prepare hover data (requires combined_faculty_df to be defined)
# --- Placeholder/Error Handling for combined_faculty_df ---
# You should replace this with your actual logic for combined_faculty_df.
try:
    combined_faculty_df
except NameError:
    st.warning("`combined_faculty_df` not found. Using placeholder data for hover text. Please define `combined_faculty_df`.")
    # Create a placeholder based on raw_data structure
    combined_faculty_df = raw_data[['Faculty_Full_Name']].copy()
    # Add dummy mesh terms or use 'Keywords' if available
    if 'Keywords' in raw_data.columns:
        combined_faculty_df['Combined_Mesh_Terms'] = raw_data['Keywords']
    else:
        combined_faculty_df['Combined_Mesh_Terms'] = [
            'TermA; TermB; TermC'] * len(combined_faculty_df)
# --- End Placeholder ---

# UMAP 2D
st.subheader("UMAP 2D Projection (on Raw Data)")  # Add subheader
# Write description for UMAP 2D Plot using st.write()
st.write("""
**Plot Description:** Shows a 2D UMAP representation of the original high-dimensional data. Aims to preserve local and global structure. Hover shows faculty name and top 3 MeSH terms (if available).
""")
umap_2d_result = UMAP(random_state=42).fit_transform(
    numeric_data)  # Add random_state
umap_2d_df = pd.DataFrame(umap_2d_result, columns=["umap_1", "umap_2"])
# Use .values for safety
umap_2d_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values

top_mesh_terms = []
# Assuming combined_faculty_df exists now
for faculty_full_name in umap_2d_df['Faculty_Full_Name']:
    try:
        faculty_terms_series = combined_faculty_df[combined_faculty_df['Faculty_Full_Name']
                                                   == faculty_full_name]['Combined_Mesh_Terms']
        if not faculty_terms_series.empty:
            faculty_terms = faculty_terms_series.iloc[0]
            if pd.isna(faculty_terms) or not isinstance(faculty_terms, str):
                top_items = ["N/A"]
            else:
                # Improved splitting and counting
                terms_list = [term.strip() for term in re.split(
                    r'[;,]\s*', faculty_terms) if term.strip()]
                counter = Counter(terms_list)
                sorted_items = sorted(
                    counter.items(), key=lambda item: (-item[1], item[0]))
                top_items = [item[0] for item in sorted_items[:3]]
        else:
            top_items = ["Not Found"]
    except Exception as e:
        # Show warning in app
        st.warning(f"Could not process terms for {faculty_full_name}: {e}")
        top_items = ["Error"]
    top_mesh_terms.append("; ".join(top_items))

umap_2d_df['Top_Mesh_Terms'] = top_mesh_terms

fig_umap2d = px.scatter(umap_2d_df, x="umap_1", y="umap_2", title="UMAP", hover_name="Faculty_Full_Name", hover_data={
                        "umap_1": False, "umap_2": False, "Top_Mesh_Terms": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])

fig_umap2d.update_layout(plot_bgcolor='#255799', title={
    'text': "UMAP",
    'font': {'size': 30},
})
fig_umap2d.update_xaxes(title_text="")
fig_umap2d.update_yaxes(title_text="")
fig_umap2d.update_xaxes(showticklabels=False)
fig_umap2d.update_yaxes(showticklabels=False)
st.plotly_chart(fig_umap2d)  # Use new variable name

# # UMAP 3D (Original code commented out)
# ...

# t-SNE
st.subheader("t-SNE 2D Projection (on Raw Data)")  # Add subheader
# Write description for t-SNE Plot using st.write()
st.write("""
**Plot Description:** Presents a 2D t-SNE embedding of the original data. Focuses on local similarities. Hover shows faculty name and top 3 MeSH terms (if available).
""")
tsne = TSNE(n_components=2, perplexity=25, random_state=42)  # Add random_state
tsne_result = tsne.fit_transform(numeric_data)
tsne_df = pd.DataFrame(tsne_result, columns=["tsne_1", "tsne_2"])
tsne_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values  # Use .values
tsne_df['Top_Mesh_Terms'] = top_mesh_terms  # Use previously calculated terms
fig_tsne = px.scatter(tsne_df, x="tsne_1", y="tsne_2", title="t-SNE", hover_name="Faculty_Full_Name", hover_data={
                      "tsne_1": False, "tsne_2": False, "Top_Mesh_Terms": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_tsne.update_layout(plot_bgcolor='#255799', title={
    'text': "t-SNE",
    'font': {'size': 30},
})
fig_tsne.update_xaxes(title_text="")
fig_tsne.update_yaxes(title_text="")
fig_tsne.update_xaxes(showticklabels=False)
fig_tsne.update_yaxes(showticklabels=False)
st.plotly_chart(fig_tsne)  # Use new variable name

# UMAP on PCA components
st.subheader("UMAP on PCA Components")  # Add subheader
num_pca_components = 2
pca_scores = pca_result[:, :num_pca_components]
umap_pca_result = UMAP(random_state=42).fit_transform(
    pca_scores)  # Add random_state
umap_pca_df = pd.DataFrame(umap_pca_result, columns=["umap_1", "umap_2"])
# Use .values
umap_pca_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values
umap_pca_df['Top_Mesh_Terms'] = top_mesh_terms  # Add hover text data
# Write description for UMAP on PCA Plot using st.write()
st.write(f"""
**Plot Description:** Displays the UMAP projection applied *after* PCA reduction (using first {num_pca_components} components). Visualizes structure found by UMAP within the principal component space. Hover shows faculty/terms.
""")
fig_umap_pca = px.scatter(umap_pca_df, x="umap_1", y="umap_2", title="UMAP on PCA Components", hover_name="Faculty_Full_Name", hover_data={
                          "umap_1": False, "umap_2": False, "Top_Mesh_Terms": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_umap_pca.update_layout(plot_bgcolor='#255799', title={
    'text': "UMAP on PCA Components",
    'font': {'size': 30},
})
fig_umap_pca.update_xaxes(title_text="")
fig_umap_pca.update_yaxes(title_text="")
fig_umap_pca.update_xaxes(showticklabels=False)
fig_umap_pca.update_yaxes(showticklabels=False)
st.plotly_chart(fig_umap_pca)  # Use new variable name

# UMAP on varying PCA components
st.subheader("UMAP on Varying PCA Dimensions")  # Add subheader
# Limit to available components
max_components_to_show = min(30, pca_result.shape[1])
for num_components in range(1, max_components_to_show + 1):
    pca_scores_var = pca_result[:, :num_components]
    umap_result = UMAP(random_state=42).fit_transform(
        pca_scores_var)  # Add random_state
    umap_df_pca_var = pd.DataFrame(umap_result, columns=["umap_1", "umap_2"])
    # Use .values
    umap_df_pca_var['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values
    umap_df_pca_var['Top_Mesh_Terms'] = top_mesh_terms  # Add hover text data
    # Write description for UMAP on Varying PCA Plot (inside loop) using st.write()
    st.write(f"""
**Plot Description (UMAP with {num_components} PCA Components):** Shows UMAP projection based on the first {num_components} principal components. Observe structure changes as more components are included. Hover shows faculty/terms.
""")
    fig_umap_pca_var = px.scatter(umap_df_pca_var, x="umap_1", y="umap_2", title=f"UMAP with {num_components} PCA Components", hover_name="Faculty_Full_Name", hover_data={
                                  "umap_1": False, "umap_2": False, "Top_Mesh_Terms": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])
    fig_umap_pca_var.update_layout(plot_bgcolor='#255799', title={
        'text': f"UMAP with {num_components} PCA Components",
        'font': {'size': 30},
    })
    fig_umap_pca_var.update_xaxes(title_text="")
    fig_umap_pca_var.update_yaxes(title_text="")
    fig_umap_pca_var.update_xaxes(showticklabels=False)
    fig_umap_pca_var.update_yaxes(showticklabels=False)
    st.plotly_chart(fig_umap_pca_var)  # Use new variable name

# DBSCAN clustering
st.subheader("UMAP with DBSCAN Clustering")  # Add subheader
# Note: Original code fits knn but doesn't use it for DBSCAN params
knn = NearestNeighbors(n_neighbors=8)
# Ensure pca_scores (from num_pca_components=2) is used
if 'pca_scores' not in locals() or pca_scores.shape[1] != 2:
    st.warning("Recalculating pca_scores for 2 components for DBSCAN/KMeans.")
    num_pca_components = 2
    pca_scores = pca_result[:, :num_pca_components]
    # Need to recalculate umap_pca_df based on these 2 components if it wasn't already
    umap_pca_result = UMAP(random_state=42).fit_transform(pca_scores)
    umap_pca_df = pd.DataFrame(umap_pca_result, columns=["umap_1", "umap_2"])
    umap_pca_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values
    umap_pca_df['Top_Mesh_Terms'] = top_mesh_terms

knn.fit(pca_scores)
distances, indices = knn.kneighbors(pca_scores)
dbscan = DBSCAN(eps=0.05, min_samples=2).fit(pca_scores)
umap_pca_df['cluster'] = dbscan.labels_  # Adds/overwrites 'cluster' column
# Write description for UMAP with DBSCAN Clusters Plot using st.write()
st.write(f"""
**Plot Description:** Shows UMAP projection (based on first {num_pca_components} PCA components). Points colored by DBSCAN clusters (eps={dbscan.eps}, min_samples={dbscan.min_samples}) found using the {num_pca_components} PCA components. Noise points often label -1. Hover shows faculty/terms/cluster. *(Note: Fixed color means all clusters appear yellow).*
""")
fig_dbscan = px.scatter(umap_pca_df, x="umap_1", y="umap_2", color='cluster', title="UMAP with DBSCAN Clusters", hover_name="Faculty_Full_Name", hover_data={
                        "umap_1": False, "umap_2": False, "Top_Mesh_Terms": True, "cluster": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_dbscan.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    title={
        'text': "UMAP with DBSCAN Clusters",  # Original Title
        'font': {'size': 30},
    })
fig_dbscan.update_xaxes(title_text="")
fig_dbscan.update_yaxes(title_text="")
fig_dbscan.update_xaxes(showticklabels=False)
fig_dbscan.update_yaxes(showticklabels=False)
st.plotly_chart(fig_dbscan)  # Use new variable name

# K-means clustering
st.subheader("UMAP with K-means Clustering (K=12)")  # Add subheader
kmeans = KMeans(n_clusters=12, random_state=123, n_init=10).fit(
    pca_scores)  # Use n_init=10 instead of 'auto' for explicit behavior
# Overwrites the 'cluster' column again
umap_pca_df['cluster'] = kmeans.labels_
# Write description for UMAP with K-means Clusters Plot using st.write()
st.write(f"""
**Plot Description:** Shows UMAP projection (based on first {num_pca_components} PCA components). Points colored by K-means clusters (K=12) found using the {num_pca_components} PCA components. Hover shows faculty/terms/cluster. *(Note: Fixed color means all clusters appear yellow).*
""")
fig_kmeans = px.scatter(umap_pca_df, x="umap_1", y="umap_2", color='cluster', title="UMAP with K-means Clusters", hover_name="Faculty_Full_Name",
                        hover_data={"umap_1": False, "umap_2": False, "Top_Mesh_Terms": True, "cluster": True}, width=800, height=800, color_discrete_sequence=['#fecc07'])
fig_kmeans.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    title={
        'text': "UMAP with K-means Clusters",  # Original Title
        'font': {'size': 30},
    })
fig_kmeans.update_xaxes(title_text="")
fig_kmeans.update_yaxes(title_text="")
fig_kmeans.update_xaxes(showticklabels=False)
fig_kmeans.update_yaxes(showticklabels=False)
st.plotly_chart(fig_kmeans)  # Use new variable name

# Silhouette score for optimal K
st.subheader("Silhouette Score for K-means")  # Add subheader
# Write description for Silhouette Score Plot using st.write()
st.write("""
**Plot Description:** Plots average silhouette score for different K values in K-means. Higher scores suggest better-defined clusters. Helps choose K. *(Note: Original code calculated this on the 2D UMAP coordinates, not the PCA data K-means was fit to, which is inconsistent.)*
""")
# WARNING: Original code calculates silhouette on umap_pca_df (2D UMAP coordinates),
# NOT on pca_scores (the data K-means was actually fit to). Sticking to original logic here.


def calculate_silhouette_score(data, k):
    # Ensure data is DataFrame before dropping column
    if isinstance(data, pd.DataFrame):
        # Drop non-numeric and cluster columns before calculating
        cols_to_drop = ['Faculty_Full_Name', 'Top_Mesh_Terms', 'cluster']
        data_numeric = data.drop(
            columns=[col for col in cols_to_drop if col in data.columns])
    else:
        data_numeric = data  # Assume numpy array

    if hasattr(data_numeric, 'shape') and data_numeric.shape[0] > 1 and data_numeric.shape[1] > 0:
        kmeans_sil = KMeans(n_clusters=k, random_state=123,
                            n_init=10).fit(data_numeric)
        labels = kmeans_sil.labels_
        if len(set(labels)) > 1:
            return silhouette_score(data_numeric, labels)
        else:
            return -1
    else:
        st.warning("Not enough data/features for Silhouette calculation.")
        return -1


k_values = range(2, 21)
# Calculate on umap_pca_df (dropping non-numeric) as per original code
silhouette_values = [calculate_silhouette_score(
    umap_pca_df.copy(), k) for k in k_values]

# Filter out invalid scores before plotting
valid_k_scores = [(k, score) for k, score in zip(
    k_values, silhouette_values) if score > -1]
if valid_k_scores:
    valid_k, valid_scores = zip(*valid_k_scores)
    plt.figure()
    plt.plot(valid_k, valid_scores, 'b*-')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Average Silhouette Width')
    plt.title('Silhouette Score for Different K')
    st.pyplot(plt.gcf())
    plt.clf()  # Add plt.clf() after st.pyplot
else:
    st.warning("Could not calculate valid Silhouette scores to plot.")


# ANOVA and feature significance
st.subheader("ANOVA Feature Significance (K=12)")  # Add subheader
# Write description for ANOVA using st.write()
st.write(f"""
**Analysis Description:** Performs ANOVA for each feature (MeSH term) across the {kmeans.n_clusters if 'kmeans' in locals() else 'N/A'} K-means clusters found previously (using clusters from K=12 on PCA data). Identifies features whose mean value differs significantly between clusters (adjusted p < 0.05). Results are printed to console/log by original code and optionally shown below.
""")
# Check if kmeans object exists
if 'kmeans' in locals() and hasattr(kmeans, 'labels_'):
    try:
        # Use original numeric data
        filtered_data = pd.DataFrame(numeric_data).copy()
        filtered_data.columns = filtered_data.columns.astype(
            str)  # Ensure string column names
        filtered_data['cluster'] = kmeans.labels_  # Labels from K=12 run

        # Clean feature names for formula usage (more robustly)
        clean_feature_names = []
        original_to_clean_map = {}
        for i, col in enumerate(numeric_data.columns):
            clean_col = str(col)  # Ensure string
            # Remove non-alphanumeric/whitespace
            clean_col = re.sub(r'[^\w\s]+', '', clean_col)
            # Replace whitespace with _, strip ends
            clean_col = re.sub(r'\s+', '_', clean_col).strip('_')
            if not clean_col:
                clean_col = f"feature_{i}"  # Handle empty names
            if clean_col[0].isdigit():
                clean_col = f"_{clean_col}"  # Handle names starting with digit
            clean_feature_names.append(clean_col)
            # Map clean back to original
            original_to_clean_map[clean_col] = col

        filtered_data.columns = clean_feature_names + ['cluster']
        feature_names = clean_feature_names

        def calculate_anova_pvalue(feature, data):
            model_formula = f"`{feature}` ~ C(cluster)"  # Use backticks
            try:
                model = ols(model_formula, data=data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                return anova_table["PR(>F)"][0]
            except Exception as e:
                # st.warning(f"ANOVA failed for feature '{feature}': {e}")
                return 1.0  # Return non-significant p-value

        p_values = {feature: calculate_anova_pvalue(
            feature, filtered_data) for feature in feature_names}
        p_values_list = list(p_values.values())

        if p_values_list:
            reject, p_adjusted, _, _ = multipletests(
                p_values_list, method='fdr_bh')
            significant_results = pd.DataFrame({
                'Clean Feature Name': feature_names,
                'Original Feature Name': [original_to_clean_map.get(f, f) for f in feature_names],
                'Adjusted P-Value (FDR)': p_adjusted,
                'Significant': reject
            })
            significant_features_df = significant_results[significant_results['Significant']].sort_values(
                'Adjusted P-Value (FDR)')

            print("--- ANOVA Significant Features (printed to console) ---")
            print(significant_features_df[[
                  'Original Feature Name', 'Adjusted P-Value (FDR)']])
            print("---")

            # Optionally display significant features in Streamlit app
            if not significant_features_df.empty:
                st.write("**Significant Features (FDR < 0.05):**")
                st.dataframe(significant_features_df[[
                             'Original Feature Name', 'Adjusted P-Value (FDR)']].reset_index(drop=True))
            else:
                st.write("No significant features found via ANOVA at FDR < 0.05.")

        else:
            st.warning("ANOVA: No valid p-values found.")

    except Exception as e:
        st.error(f"An error occurred during the ANOVA analysis: {e}")
else:
    st.warning("Skipping ANOVA: K-means results not available.")


# Final UMAP with K-means clusters
st.subheader("Final UMAP Visualization (K=12, Black)")  # Add subheader
# Write description for Final UMAP Plot using st.write()
st.write(f"""
**Plot Description:** Another view of UMAP (based on {num_pca_components} PCA components) colored by the K=12 K-means clusters. Hover shows faculty/terms/cluster. *(Note: Fixed BLACK color means all clusters appear black).*
""")
# Ensure 'cluster' column from K-means exists
if 'cluster' in umap_pca_df.columns and 'kmeans' in locals():
    # Ensure column is present
    umap_pca_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name'].values
    umap_pca_df['Top_Mesh_Terms'] = top_mesh_terms  # Ensure column is present
    fig_final = px.scatter(umap_pca_df, x="umap_1", y="umap_2", color='cluster', title="UMAP with K-means Clusters", hover_name="Faculty_Full_Name", hover_data={
                           "umap_1": False, "umap_2": False, "Top_Mesh_Terms": True, "cluster": True}, width=800, height=800, color_discrete_sequence=['#000000'])  # Original black color
    fig_final.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        title={
            'text': "UMAP with K-means Clusters",  # Original Title
            'font': {'size': 30},
        })
    fig_final.update_xaxes(title_text="")
    fig_final.update_yaxes(title_text="")
    fig_final.update_xaxes(showticklabels=False)
    fig_final.update_yaxes(showticklabels=False)
    # Original call, use new variable name
    st.plotly_chart(fig_final, key="final_umap_cluster_plot")
else:
    st.warning("Cannot display final K-means plot - clustering results missing.")

# Save outputs
# output_path = 'faculty-mapped-mesh-terms'
# significant_features_path = os.path.join(output_path, "Significant_terms_per_cluster.csv")
# professors_clusters_path = os.path.join(output_path, "Professors_in_clusters.csv")

# umap_pca_df_grouped = umap_pca_df.drop(columns=['umap_1', 'umap_2'])
# umap_pca_df_grouped = umap_pca_df_grouped.groupby('cluster')['Faculty_Full_Name'].apply(list).reset_index()
# umap_pca_df_grouped.to_csv(professors_clusters_path, index=True)

# significant_df = pd.DataFrame({
#     'Feature': significant_features,
#     'P-Value Adjusted': p_adjusted[:len(significant_features)]
# })
# uncomment when not running streamlit
# significant_df.to_csv(significant_features_path, index=True)

st.info("Script execution finished.")  # Final message in the app