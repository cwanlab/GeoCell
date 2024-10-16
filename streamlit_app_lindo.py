import altair as alt
import pandas as pd
import streamlit as st


# ========================================================================
# @Discripiton: GeoCells
# streamlit run streamlit_app.py from the terminal at ANACONDA
# publish:https://cwanlab-geocell-streamlit-app-vuwi9l.streamlit.app/
# @authors: Melnikas, Max; Nkambule, Lethukuthula; Wan, Guihong
# @date: Oct 12, 2024
# ========================================================================

st.header("BMI 706: Final Project", divider=False)
st.text("Melnikas Max, Nkambule Lethukuthula, Wan Guihong")
st.divider()

# ==============================================
df = pd.read_parquet('data/cell_spatial_distribution.parquet', engine='fastparquet')
# "https://raw.githubusercontent.com/cwanlab/GeoCell/main/data/cell_spatial_distribution.parquet"

# Data summary
# Percentage Bar
count_data = df['phenotype'].value_counts(normalize=True).reset_index()
count_data.columns = ['phenotype', 'Percentage']
count_data['Percentage'] *= 100  #percentage

# use header instead of title
percentages = alt.Chart(count_data).mark_bar().encode(
    y=alt.Y("phenotype:N", title="Phenotype", sort="-x"),
    x=alt.X("Percentage:Q", title="Percentage (%)"),
    # color=alt.Color("phenotype:N"),
    tooltip=["phenotype:N", "Percentage:Q"]
).properties(
    width=800,
    title="Percentage of Cells by Phenotype"
).configure_title(
    fontSize=18
).configure_axis(
    titleFontSize=14
).configure_legend(
    titleFontSize=14
)

st.altair_chart(percentages, use_container_width=True)
st.divider()

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
    bind=alt.binding_select(
        options=[None] + options,
        labels = ['All'] + labels,
        name='Phenotype: '
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
st.divider()

# ==============================================
# Clustering visualization
# ==============================================

combined_df = pd.read_parquet('data/umap_tsne_combined_data.parquet', engine='fastparquet')
# "https://raw.githubusercontent.com/cwanlab/GeoCell/main/data/umap_tsne_combined_data.parquet"

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
    bind=alt.binding_select(
        options=['UMAP', 'TSNE'],
        name='Projection: '
    ),
    name='Projection',
    value = 'UMAP'
)
# Clustering method selection
clustering_select = alt.selection_point(
    fields=['cluster_type'],
    bind=alt.binding_select(
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
    bind=alt.binding_select(
        options=[None] + options_cluster,
        labels = ['All'] + labels_cluster,
        name='Cluster: '
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
st.altair_chart(combined_chart, use_container_width=True)
st.divider()
