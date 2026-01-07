import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Page config
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("Hierarchical Clustering Visualization App")

# Load saved model & scaler
@st.cache_resource
def load_objects():
    model = joblib.load("hirarchical_clustering/hierarchical_clustering_model.pkl")
    scaler = joblib.load("hirarchical_clustering/scaler.pkl")
    return model, scaler

model, scaler = load_objects()

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        st.error("Dataset must contain at least 2 numerical columns")
    else:
        # Scale data
        X_scaled = scaler.transform(numeric_df)

        # Dendrogram
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        linkage_matrix = linkage(X_scaled, method="ward")
        dendrogram(linkage_matrix, ax=ax)
        st.pyplot(fig)

        # Apply clustering (re-fit for visualization)
        clusters = model.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.subheader("Clustered Dataset")
        st.dataframe(df.head())

        # Scatter plot
        st.subheader("Cluster Visualization")
        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=clusters
        )
        ax2.set_xlabel(numeric_df.columns[0])
        ax2.set_ylabel(numeric_df.columns[1])
        ax2.set_title("Hierarchical Clustering Result")
        st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to start clustering.")
