import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

st.title("K-Means Clustering untuk Data Obesitas dan Risiko CVD")

# Load dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Awal")
    st.write(df.head())

    st.subheader("Informasi Data")
    buffer = df.info(buf=None)
    st.text(buffer)

    # Preprocessing
    selected_columns = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','NObeyesdad']
    df = df[selected_columns]

    # Label Encoding
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.drop('NObeyesdad', axis=1))

    # Elbow Method
    st.subheader("Menentukan Jumlah Cluster dengan Metode Siku (Elbow)")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Metode Siku")
    ax.set_xlabel("Jumlah Klaster")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # Pemodelan dengan k = 3
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels

    # Evaluasi
    silhouette = silhouette_score(X_scaled, cluster_labels)
    dbi = davies_bouldin_score(X_scaled, cluster_labels)

    st.subheader("Evaluasi Model Clustering")
    st.write(f"Silhouette Score: {silhouette:.3f}")
    st.write(f"Davies-Bouldin Index: {dbi:.3f}")

    # Tampilkan data yang sudah diberi cluster
    st.subheader("Data dengan Label Klaster")
    st.write(df.head())
