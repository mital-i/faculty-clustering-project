import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
import re
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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from streamlit_plotly_events import plotly_events
import networkx as nx
import leidenalg as la
import igraph as ig
from sklearn.preprocessing import StandardScaler

import streamlit as st
print(st.__version__)

# Configuration
config = {
    'file_path': 'mesh_terms_matrix_5yrs_and_keywords.xlsx',
    'pca_components_to_try': range(1, 7),
    'dbscan_eps': 0.05,
    'dbscan_min_samples': 2,
    'anova_alpha': 0.05,
    'top_n_features_to_plot': 10,
    'cluster_output_path': 'Professors_in_clusters.csv',
    'anova_output_path': 'Significant_terms_per_cluster.csv',
    'top_mesh_terms_output_path': 'Top_Mesh_Terms_Per_Professor.csv',
}

color_palette = px.colors.qualitative.Set1

# Helper functions


def load_and_preprocess_data(file_path, index_col='Faculty_Full_Name'):
    raw_data = pd.read_excel(file_path, index_col=index_col)
    raw_data.reset_index(inplace=True)
    faculty_names_df = pd.read_excel(file_path, usecols=['Faculty_Full_Name'])
    feature_matrix = raw_data.drop(columns=['Faculty_Full_Name'])
    raw_data.columns = raw_data.columns.str.replace(
        ' ', '_').str.replace('-', '_').str.replace(',', '_')
    feature_matrix.columns = feature_matrix.columns.str.replace(
        ' ', '_').str.replace('-', '_').str.replace(',', '_')
    return raw_data, feature_matrix, faculty_names_df


def format_mesh_terms(mesh_terms):
    if isinstance(mesh_terms, list) and len(mesh_terms) > 0 and isinstance(mesh_terms[0], list):
        mesh_terms = mesh_terms[0]
    return ', '.join([term.replace('_', ' ') for term in mesh_terms[:5] if isinstance(term, str)])


# Load and preprocess data
raw_data, feature_matrix, faculty_names_df = load_and_preprocess_data(
    config['file_path'])
mesh_term_columns = [col for col in feature_matrix.columns]

# Calculate top MeSH terms
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
    top_mesh_terms_list.append([top_3_terms])
    faculty_names.append(professor_name)

top_mesh_terms_df = pd.DataFrame(
    {'Faculty_Full_Name': faculty_names, 'Top_Mesh_Terms': top_mesh_terms_list})
top_mesh_terms_df.set_index('Faculty_Full_Name', inplace=True)
top_mesh_terms_df.to_csv(config['top_mesh_terms_output_path'])

# PCA
pca = PCA()
pca_embeddings = pca.fit_transform(feature_matrix)

# Streamlit app
st.title("Faculty Research Analysis")

# PCA Explained Variance
st.subheader("PCA Explained Variance")
st.write("""
**Plot Description:** This plot shows the cumulative percentage of variance captured by the principal components (PCs).
The x-axis represents the number of PCs included, and the y-axis shows the total variance explained.
It helps determine how many PCs are needed to retain a significant amount of information.
""")
explained_variance = pca.explained_variance_ratio_
fig, ax = plt.subplots()
ax.plot(np.cumsum(explained_variance))
ax.set_xlabel('Number of Components')
ax.set_ylabel('Variance (%)')
ax.set_title('Explained Variance')
st.pyplot(fig)

# PCA Scatter Plot
st.subheader("PCA Scatter Plot (PC1 vs PC2)")
st.write("""
**Plot Description:** This scatter plot displays the data projected onto the first two principal components (PC1 and PC2).
Each point represents a faculty member. It reveals the primary axes of variation based on PCA reduction.
""")
pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[
                                 f'PC{i+1}' for i in range(pca_embeddings.shape[1])])
pca_embeddings_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
pca_embeddings_df = pca_embeddings_df.merge(
    top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(pca_embeddings_df, x='PC1', y='PC2', hover_name='Faculty_Full_Name',
                 hover_data={'PC1': False, 'PC2': False, 'Top_Mesh_Terms': True})
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=pca_embeddings_df['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

# UMAP 2D
st.subheader("UMAP 2D Projection")
st.write("""
**Plot Description:** This plot shows a 2D UMAP representation of the original high-dimensional data.
It aims to preserve both local and global structure. Hover over points to see faculty names and top MeSH terms.
""")
umap_embeddings = UMAP().fit_transform(feature_matrix)
umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=["V1", "V2"])
umap_embeddings_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_embeddings_df = umap_embeddings_df.merge(
    top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_embeddings_df, x="V1", y="V2", title="UMAP", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_embeddings_df['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

# UMAP with PCA components
st.subheader("UMAP on PCA Components")
num_components = st.slider("Number of PCA components",
                           min_value=1, max_value=10, value=3)
pca_reduced_features = pca_embeddings[:, :num_components]
umap_result = UMAP(random_state=123).fit_transform(pca_reduced_features)
umap_df_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_df_pca = umap_df_pca.merge(
    top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_df_pca, x="V1", y="V2", title=f"UMAP on {num_components} PCA Components", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_df_pca['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

# Clustering
st.subheader("Clustering")
clustering_method = st.selectbox("Select clustering method", [
                                 "K-means", "DBSCAN", "Leiden"])

if clustering_method == "K-means":
    n_clusters = st.slider("Number of clusters",
                           min_value=2, max_value=20, value=8)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(umap_result)
    # Ensure cluster labels are assigned here
    umap_df_pca['cluster'] = cluster_labels
elif clustering_method == "DBSCAN":
    eps = st.slider("DBSCAN eps", min_value=0.01,
                    max_value=1.0, value=0.05, step=0.01)
    min_samples = st.slider("DBSCAN min_samples",
                            min_value=2, max_value=10, value=2)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(umap_result)
    # Ensure cluster labels are assigned here
    umap_df_pca['cluster'] = cluster_labels
else:  # Leiden
    resolution = st.slider("Leiden resolution", min_value=0.1,
                           max_value=2.0, value=0.8, step=0.1)
    knn = NearestNeighbors(n_neighbors=15)
    knn.fit(umap_result)
    _, indices = knn.kneighbors(umap_result)
    graph = nx.Graph()
    for i in range(len(umap_result)):
        for j in indices[i]:
            if i != j:
                graph.add_edge(i, j)
    g_ig = ig.Graph.from_networkx(graph)
    partition = la.find_partition(
        g_ig, la.CPMVertexPartition, resolution_parameter=resolution)
    # Ensure cluster labels are assigned here
    umap_df_pca['cluster'] = partition.membership

fig = px.scatter(umap_df_pca, x="V1", y="V2", color='cluster', title=f"UMAP with {clustering_method} Clustering",
                 hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_df_pca['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

# Add search box
search_term = st.text_input("Search for faculty name or MeSH term:")

# Create a mask for filtering faculty names and MeSH terms
if search_term:
    faculty_mask = umap_df_pca['Faculty_Full_Name'].str.contains(
        search_term, case=False)
    mesh_mask = umap_df_pca['Top_Mesh_Terms'].apply(
        lambda x: any(search_term.lower() in term.lower() for term in x[0]))
    filtered_df = umap_df_pca[faculty_mask | mesh_mask]
else:
    filtered_df = umap_df_pca

# Create a scatter plot with highlighted faculty
fig = px.scatter(umap_df_pca, x="V1", y="V2", color='cluster',
                 color_discrete_sequence=color_palette, title=f"UMAP with {clustering_method} Clustering",
                 hover_name="Faculty_Full_Name", hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)

# Highlight selected faculty in red
if search_term:
    for index, row in filtered_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['V1']],
            y=[row['V2']],
            mode='markers',
            marker=dict(color='red', size=10),
            name=row['Faculty_Full_Name'],
            hoverinfo='text',
            text=f"{row['Faculty_Full_Name']}<br>Top MeSH Terms: {format_mesh_terms(row['Top_Mesh_Terms'])}",
            showlegend=False
        ))

# Update hover template to include cluster information
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_df_pca['Top_Mesh_Terms'].apply(format_mesh_terms)
)

# # Display the plot
# plot_placeholder = st.empty()
# plot_placeholder.plotly_chart(fig)

# # Handle click events to unselect faculty
# clicked_point = st.experimental_get_query_params().get('clickData')
# if clicked_point:
#     point_data = clicked_point[0]
#     faculty_name = point_data.get('points', [{}])[0].get('hovertext')
#     if faculty_name:
#         search_term = ""  # Clear the search term to unselect
#         st.experimental_set_query_params()

# # Display selected faculty information

# # Load the Excel file
# faculty_mesh_terms_df = pd.read_excel("faculty_unique_mesh_terms.xlsx")

# faculty_mesh_terms_df.columns = faculty_mesh_terms_df.columns.str.strip()
# if "Faculty" in faculty_mesh_terms_df.columns:
#     faculty_mesh_terms_df.rename(
#         columns={"Faculty": "Faculty_Full_Name"}, inplace=True
#     )

# # Normalised key column → always lowercase, single-spaced, trimmed
# faculty_mesh_terms_df["name_key"] = (
#     faculty_mesh_terms_df["Faculty_Full_Name"]
#     .str.replace(r"\s+", " ", regex=True)
#     .str.strip()
#     .str.casefold()
# )

# if search_term:
#     selected_faculty_info = filtered_df[
#         filtered_df["Faculty_Full_Name"].str.contains(search_term, case=False)
#     ]

#     if not selected_faculty_info.empty:
#         st.sidebar.subheader("Selected Faculty Information")

#         for _, row in selected_faculty_info.iterrows():
#             prof = row["Faculty_Full_Name"]
#             prof_key = re.sub(r"\s+", " ", prof).strip().casefold()

#             # Match on the normalised key, not the raw string
#             full_terms_row = faculty_mesh_terms_df[
#                 faculty_mesh_terms_df["name_key"] == prof_key
#             ]

#             if full_terms_row.empty:
#                 st.sidebar.write(
#                     f"**{prof}** — no entry in faculty_unique_mesh_terms.xlsx")
#                 continue

#             mesh_terms = full_terms_row.iloc[0]["Unique_Mesh_Terms"]

#             # Convert "term1; term2" → list if necessary
#             if isinstance(mesh_terms, str):
#                 mesh_terms = [t.strip() for t in re.split(
#                     r"[;,]", mesh_terms) if t.strip()]

#             # ---------- sidebar display ----------
#             st.sidebar.markdown(f"**{prof}**")
#             st.sidebar.write(f"Unique MeSH terms ({len(mesh_terms)}):")
#             st.sidebar.write(", ".join(mesh_terms))

#             # ---------- download button ----------
#             csv_bytes = (
#                 pd.DataFrame({"MeSH_Term": mesh_terms})
#                 .to_csv(index=False)
#                 .encode("utf-8")
#             )
#             st.sidebar.download_button(
#                 label="Download full MeSH-term list as CSV",
#                 data=csv_bytes,
#                 file_name=f"{prof.replace(' ', '_')}_mesh_terms.csv",
#                 mime="text/csv",
#             )
fig.update_traces(
    # customdata must be 2-D; wrap each name in a list
    customdata=umap_df_pca["Faculty_Full_Name"].apply(lambda x: [x]).tolist(),
    hovertemplate='<b>%{customdata[0]}</b><br><br>'
                  'Top Keywords: %{customdata[1]}<extra></extra>'
)
# Note: the 2nd customdata field gets added automatically by the block that
# formats Top_Mesh_Terms earlier in your script.

# ---------------------------------------------------------------
# 1) Load the unique-MeSH-term lookup *once*  (place near top)
# ---------------------------------------------------------------
faculty_mesh_terms_df = (
    pd.read_excel("faculty_unique_mesh_terms.xlsx")
      .rename(columns=lambda c: c.strip())
)
if "Faculty" in faculty_mesh_terms_df.columns:
    faculty_mesh_terms_df.rename(
        columns={"Faculty": "Faculty_Full_Name"}, inplace=True)

# ---------------------------------------------------------------
# 2) Build a helper column that holds a pretty one-line term list
# ---------------------------------------------------------------
faculty_mesh_terms_df["term_string"] = (
    faculty_mesh_terms_df["Unique_Mesh_Terms"]
    .apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
)

# ---------------------------------------------------------------
# 3) Before **any** update_traces() that will feed Plotly-events,
#    make sure each point carries *both* pieces of info
# ---------------------------------------------------------------
tooltip_terms = umap_df_pca["Top_Mesh_Terms"].apply(format_mesh_terms)

fig.update_traces(
    customdata=np.column_stack(
        (umap_df_pca["Faculty_Full_Name"], tooltip_terms)
    ).tolist(),
    hovertemplate='<b>%{customdata[0]}</b><br><br>'
                  'Top MeSH terms: %{customdata[1]}<extra></extra>'
)


# --- 2. Plotly events: click OR box/lasso select -----------------------------
selected_points = plotly_events(
    fig,
    click_event=True,    # single clicks add/remove a point
    select_event=True,   # box/lasso adds all points in selection
    override_height=800, override_width="100%"
)

# Keep a running list in st.session_state so multiple interactions accumulate
if "picked" not in st.session_state:
    st.session_state.picked = set()

for p in selected_points:
    if "customdata" in p and p["customdata"]:
        name_clicked = p["customdata"][0]           # preferred path
    else:
        # Fallback: grab the name directly from the dataframe
        name_clicked = umap_df_pca.iloc[p["pointIndex"]]["Faculty_Full_Name"]
    if name_clicked in st.session_state.picked:
        st.session_state.picked.remove(
            name_clicked)   # second click → unselect
    else:
        st.session_state.picked.add(name_clicked)

# Clear-selection button
if st.session_state.picked:
    if st.button("Clear all selections"):
        st.session_state.picked.clear()

# --- 3. Layout: two columns --------------------------------------------------
info_placeholder = st.container()

with info_placeholder:
    st.markdown("### Selected Faculty")
    if st.session_state.picked:
        # a) Build dataframe with full MeSH terms for each selected name
        sel_df = faculty_mesh_terms_df[
            faculty_mesh_terms_df["Faculty_Full_Name"]
            .str.casefold()
            .isin([n.casefold() for n in st.session_state.picked])
        ][["Faculty_Full_Name", "Unique_Mesh_Terms"]].copy()

        # b) Normalise the MeSH-term column into lists
        def to_list(term_field):
            if isinstance(term_field, list):
                return term_field
            return [t.strip() for t in re.split(r"[;,]", str(term_field)) if t.strip()]

        sel_df["Unique_Mesh_Terms"] = sel_df["Unique_Mesh_Terms"].apply(
            to_list)

        # c) Display
        for _, r in sel_df.iterrows():
            st.write(f"**{r['Faculty_Full_Name']}**  \n"
                     f"{', '.join(r['Unique_Mesh_Terms'])}")

        # d) Download button – one row per faculty, terms semicolon-joined
        csv_bytes = (
            sel_df.assign(
                Unique_Mesh_Terms=sel_df.Unique_Mesh_Terms.apply("; ".join)
            )
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            "Download selected faculty + MeSH terms",
            data=csv_bytes,
            file_name="selected_faculty_mesh_terms.csv",
            mime="text/csv",
        )
    else:
        st.info("Click or lasso points in the plot to see details here.")

st.plotly_chart(fig, use_container_width=True)

# ANOVA and feature significance
st.subheader("Feature Significance Analysis")
filtered_data_df = pd.DataFrame(feature_matrix)
filtered_data_df['cluster'] = umap_df_pca['cluster']
cluster_feature_matrix = filtered_data_df
feature_names = [
    col for col in cluster_feature_matrix.columns if col != 'cluster']


def get_anova_pvalues(feature, data):
    model = ols(f"{feature} ~ C(cluster)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table["PR(>F)"][0]


feature_p_values = {}
for feature in feature_names:
    try:
        feature_p_values[feature] = get_anova_pvalues(
            feature, cluster_feature_matrix)
    except:
        print(f"Error processing feature: {feature}")
        feature_p_values[feature] = 1.0

feature_list = list(feature_p_values.keys())
pval_list = list(feature_p_values.values())
_, adjusted_p_values, _, _ = multipletests(
    pval_list, method='fdr_bh', alpha=0.05)

results_df = pd.DataFrame({
    'Feature': feature_list,
    'p_value': pval_list,
    'adjusted_p_values': adjusted_p_values,
    'significant': adjusted_p_values < 0.05
})

results_df = results_df.sort_values('adjusted_p_values')
significant_features_df = results_df[results_df['significant']]
st.write(f"Found {len(significant_features_df)} significant features:")
st.write(significant_features_df[['Feature', 'adjusted_p_values']])

# Visualize top significant features
st.subheader("Top Significant Features")
top_n = st.slider("Number of top features to display",
                  min_value=1, max_value=20, value=10)


def plot_top_features(cluster_feature_matrix, significant_features_df, top_n=10):
    top_features = significant_features_df['Feature'].head(top_n).tolist()

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        sns.boxplot(x='cluster', y=feature,
                    data=cluster_feature_matrix, ax=axes[i])
        axes[i].set_title(f"{feature.split('Normalized_')[-1][:15]}...")
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    plt.tight_layout()
    return fig


if len(significant_features_df) > 0:
    fig = plot_top_features(cluster_feature_matrix,
                            significant_features_df, top_n)
    st.pyplot(fig)

# Save outputs
umap_df_pca.to_csv(config['cluster_output_path'], index=False)
significant_features_df.to_csv(config['anova_output_path'], index=False)

st.success("Analysis complete! Output files have been saved.")
