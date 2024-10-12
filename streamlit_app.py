import altair as alt
import pandas as pd
import streamlit as st
import anndata as ad
import numpy as np
import requests

# ========================================================================
# @Discripiton: GeoCells
# streamlit run streamlit_app.py from the terminal at ANACONDA
# publish:https://cwanlab-geocell-streamlit-app-vuwi9l.streamlit.app/
# @authors: Melnikas, Max; Nkambule, Lethukuthula; Wan, Guihong
# @date: Oct 12, 2024
# ========================================================================

@st.cache_data
def load_data():
    url = "https://www.dropbox.com/scl/fi/5ggx19u6153cza8jndkdp/data_final.h5ad?rlkey=17d9dyfuc1deg2argtmo1rv0l&dl=1"
    local_path = "data_final.h5ad"
    
    # Download the file
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)
    
    # Load the data
    adata = ad.read_h5ad(local_path)
    return adata

adata = load_data()

# ==============================================
# Preparing the data
df = adata.obs[['X_centroid', 'Y_centroid', 'phenotype']]
df = df.reset_index() # Convert the relevant data to a DataFrame

# Base
base = alt.Chart(df).mark_circle(size=30, opacity=0.7).encode(
    x='X_centroid:Q',
    y=alt.Y('Y_centroid:Q', scale=alt.Scale(domain=[df['Y_centroid'].max(), df['Y_centroid'].min()])),  # Flip the y-axis
    color='phenotype:N',
    tooltip=['X_centroid', 'Y_centroid', 'phenotype']
).properties(
    width=800,
    height=500,
    title='Cell Spatial Distribution by Phenotype'
)

# Ratio Highlighting
options = sorted(df['phenotype'].unique().tolist())
labels = [option + ' ' for option in options]
radio_select = alt.selection_point(
    fields=['phenotype'],
    bind=alt.binding_radio(
        options=[None] + options,
        labels = ['All'] + labels,
        name='Select Phenotype: '
    )
)
phenotype_color_condition = alt.condition(
    radio_select,
    alt.Color('phenotype:N'),
    alt.value('lightgray')
)
ratio_highlight = base.add_params(
    radio_select
).encode(
    color=phenotype_color_condition
)

st.altair_chart(ratio_highlight, use_container_width=True)

# Percentage Bar
count_data = df['phenotype'].value_counts(normalize=True).reset_index()
count_data.columns = ['phenotype', 'Percentage']
count_data['Percentage'] *= 100  #percentage

percentages = alt.Chart(count_data).mark_bar().encode(
    x=alt.X("phenotype:N", sort=options),
    y=alt.Y("Percentage:Q", title="Percentage (%)"),
    color=alt.Color("phenotype:N"),
    tooltip=["phenotype:N", "Percentage:Q"]
).properties(
    width=800,
    title="Percentage of Cells by Phenotype"
)
st.altair_chart(percentages, use_container_width=True)



# ==============================================
# Clustering visualization
# ==============================================

def normalize(df, dim1_col, dim2_col):
    df[dim1_col] = (df[dim1_col] - df[dim1_col].min()) / (df[dim1_col].max() - df[dim1_col].min())
    df[dim2_col] = (df[dim2_col] - df[dim2_col].min()) / (df[dim2_col].max() - df[dim2_col].min())
    return df

umap_df = pd.DataFrame(adata.obsm['umap'], columns=['Dim1', 'Dim2'], index=adata.obs.index)
umap_df = normalize(umap_df, 'Dim1', 'Dim2')
umap_df['leiden'] = adata.obs['leiden']
umap_df['kmeans'] = adata.obs['kmeans']
umap_df['type'] = 'UMAP'
umap_df['X_centroid'] = adata.obs['X_centroid']
umap_df['Y_centroid'] = adata.obs['Y_centroid']

tsne_df = pd.DataFrame(adata.obsm['X_tsne'], columns=['Dim1', 'Dim2'], index=adata.obs.index)
tsne_df = normalize(tsne_df, 'Dim1', 'Dim2')
tsne_df['leiden'] = adata.obs['leiden']
tsne_df['kmeans'] = adata.obs['kmeans']
tsne_df['type'] = 'TSNE'
tsne_df['X_centroid'] = adata.obs['X_centroid']
tsne_df['Y_centroid'] = adata.obs['Y_centroid']

combined_df1 = pd.concat([umap_df, tsne_df]).reset_index()
combined_df = combined_df1.melt(id_vars=['Dim1', 'Dim2', 'type', 'X_centroid', 'Y_centroid'], 
                               value_vars=['leiden', 'kmeans'],
                               var_name='cluster_type', 
                               value_name='cluster')

chart = alt.Chart(combined_df).mark_circle(size=30, opacity=0.7).encode(
    x=alt.X('Dim1:Q', title='Dimension 1'),
    y=alt.Y('Dim2:Q', title='Dimension 2'),
    tooltip=['Dim1', 'Dim2', 'cluster']
).properties(
    width=500,
    height=300,
    title='Visualization with Clusters'
).interactive()

# Projection selection
projection_select = alt.selection_point(
    fields=['type'],
    bind=alt.binding_radio(
        options=['UMAP', 'TSNE'],
        name='Projection: '
    ),
    name='Projection',
    value = 'UMAP'
)
# Clustering method selection
clustering_select = alt.selection_point(
    fields=['cluster_type'],
    bind=alt.binding_radio(
        options=['leiden', 'kmeans'],
        labels=['Leiden', 'KMeans'],
        name='Clustering: '
    ),
    name='Clustering',
    value = 'leiden'
)

# Cluster selection
options_cluster = sorted(combined_df['cluster'].unique().tolist())
labels_cluster = [option + ' ' for option in options_cluster]
radio_select_cluster = alt.selection_point(
    fields=['cluster'],
    bind=alt.binding_radio(
        options=[None] + options_cluster,
        labels = ['All'] + labels_cluster,
        name='Select cluster: '
    )
)
cluster_color_condition = alt.condition(
    radio_select_cluster,
    alt.Color('cluster:N'),
    alt.value('lightgray')
)

### chart 1
cluster_ratio_highlight = chart.add_params(
    projection_select,
    clustering_select,
    radio_select_cluster
).transform_filter(
    projection_select & clustering_select 
).encode(
    color=cluster_color_condition
)

### chart 2
cluster_chart = alt.Chart(combined_df).mark_circle(size=30, opacity=0.7).encode(
    x='X_centroid:Q',
    y=alt.Y('Y_centroid:Q', scale=alt.Scale(domain=[combined_df['Y_centroid'].max(), combined_df['Y_centroid'].min()])),
    color='cluster:N',
    tooltip=['X_centroid', 'Y_centroid', 'cluster']
).properties(
    width=500,
    height=300,
    title='Cell Spatial Distribution by Cluster'
).add_params(
    clustering_select,
    radio_select_cluster
).transform_filter(
     clustering_select 
).encode(
    color=cluster_color_condition
)

combined_chart = alt.hconcat(cluster_ratio_highlight, cluster_chart)

# alt.hconcat(cluster_ratio_highlight, cluster_chart)
# st.altair_chart(cluster_ratio_highlight, use_container_width=True)
st.altair_chart(combined_chart, use_container_width=True)

# # Heatmap: Unfinished
# phe_mask = adata.obs['phenotype'] == radio_select
# phe_adata = adata[phe_mask]
# markers = ['HHLA2', 'CMA1', 'SOX10', 'S100B', 'KERATIN', 'CD1A', 'CD163', 'CD3D',
#        'C8A', 'MITF', 'FOXP3', 'PDL1', 'KI67', 'LAG3', 'TIM3', 'PCNA',
#        'pSTAT1', 'cPARP', 'SNAIL', 'aSMA', 'HLADPB1', 'S100A', 'CD11C', 'PD1',
#        'LDH', 'PANCK', 'CCNA2', 'CCND1', 'CD63', 'CD31']

# adata_df = pd.DataFrame(
#     phe_adata.X,
#     columns=phe_adata.var.index 
# )
# adata_df.index = phe_adata.obs.index 

# heatmap_data = adata_df.reset_index().melt(id_vars='index', var_name='Marker', value_name='Expression')

# heatmap = alt.Chart(heatmap_data).mark_rect().encode(
#     x=alt.X('Marker:N', title='Marker', sort=alt.SortField('Marker', order='ascending')),
#     y=alt.Y('index:N', title='Observation', sort=alt.SortField('index', order='ascending')),
#     color=alt.Color('Expression:Q', scale=alt.Scale(scheme='viridis'), title='Expression Level'),
#     tooltip=['index', 'Marker', 'Expression']
# ).properties(
#     title="Expression Heatmap",
#     width=800,
#     height=400
# )

# st.altair_chart(heatmap, use_container_width=True)
