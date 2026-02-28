import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
print("--- Customer Segmentation Data Preview ---")
print(data.head())
plt.savefig("segmentation_plot.png")
print("\nSegmentation plot saved to 'segmentation_plot.png'.")
