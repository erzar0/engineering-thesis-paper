clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=1
).fit_transform(e_xy_flat_normalized)

clusterable_embedding = clusterable_embedding.reshape(e_xy_flat.shape[0], 1)
labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=2500,
    metric='l2'
).fit_predict(clusterable_embedding) + 1