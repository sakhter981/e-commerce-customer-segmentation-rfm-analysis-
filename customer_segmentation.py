import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Data (Simulated E-commerce Transactions)
data = {
    'CustomerID': np.arange(1, 501),
    'Recency': np.random.randint(1, 365, 500),    # Days since last purchase
    'Frequency': np.random.randint(1, 50, 500),   # Number of purchases
    'Monetary': np.random.randint(50, 5000, 500)  # Total spend
}
df = pd.DataFrame(data)

# 2. Preprocessing & Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# 3. K-Means Clustering (AI Model)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 4. Labeling Segments for Business
def label_clusters(cluster):
    if cluster == 0: return 'Champions (Best Customers)'
    elif cluster == 1: return 'At Risk (Haven\'t shopped in a while)'
    elif cluster == 2: return 'Loyal Customers (Shop often)'
    else: return 'Big Spenders (High Value)'

df['Segment'] = df['Cluster'].apply(label_clusters)

print(df.groupby('Segment').mean())
print("\nSegmentation complete. Exporting to CSV for Power BI...")
df.to_csv('segmented_customers.csv', index=False)