# PDS-08-09

# Customer Segmentation Using K-Means Clustering

## Project Overview
This project applies **K-Means Clustering** to segment customers based on their spending habits using the Mall Customer Segmentation dataset from Kaggle. The aim is to divide customers into groups (clusters) that exhibit similar characteristics, which can be useful for businesses to target specific customer segments.

## Dataset
The dataset used is the **Mall Customer Segmentation Dataset** from Kaggle, which contains the following features:
- `CustomerID`: Unique customer identifier.
- `Gender`: Customer's gender.
- `Age`: Customer's age.
- `Annual Income (k$)`: Annual income of the customer in thousands of dollars.
- `Spending Score (1-100)`: A score assigned to the customer based on their spending behavior.

## Project Objective
The main goal is to cluster customers based on their spending habits, enabling businesses to identify different segments of customers. These clusters can be used for targeted marketing strategies and personalized offerings.

## Project Workflow

### 1. **Preprocessing**
- **Scaling numerical features**: To normalize the features (`Age`, `Annual Income`, `Spending Score`), we use **MinMaxScaler**. This ensures that all features are scaled to the same range, improving the performance of the K-Means algorithm.

### 2. **Visualization**
- **Elbow method**: We use the Elbow method to determine the optimal number of clusters (K). The Elbow method involves plotting the within-cluster sum of squares (WCSS) for various values of K, and identifying the point where the rate of decrease slows down (the "elbow").
  
### 3. **Modeling**
- **K-Means Clustering**: We apply K-Means clustering with the optimal K determined by the Elbow method to group customers into clusters. 
- **Cluster Visualization**: Scatter plots are used to visualize the clusters based on different features, such as `Annual Income` and `Spending Score`.

### 4. **Evaluation**
- **Silhouette Score**: We evaluate the clustering performance using the **Silhouette Score**. Since K-Means is unsupervised, traditional metrics like accuracy and precision-recall are not applicable. The Silhouette Score measures how well-separated and cohesive the clusters are.

## Code Example

Here’s a snippet of the key code used to apply the K-Means algorithm:

```python
# Scaling numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-Means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualizing clusters
import seaborn as sns
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue=y_kmeans, data=data)
