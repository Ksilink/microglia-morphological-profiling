#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg



def create_directories(base_path):
    """Create necessary directories for saving outputs."""
    paths = {
        'params_path': os.path.join(base_path, 'params'),
        'QC_figure_path': os.path.join(base_path, 'QC_figures'),
        'base_path_csv': os.path.join(base_path, 'csv'),
        'final_figure_path' : os.path.join(base_path, 'results_figures')
        
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def run_pca(data_features, ind_train, ind_val, cols, n_pcs, seed):
    """Run PCA on training and test data."""
    pca = PCA(n_components=n_pcs, random_state=seed)
    pca_pip = Pipeline(steps=[("pca", pca)])
    
    # Train PCA
    pca_train = pca_pip.fit_transform(data_features.loc[ind_train, cols])
    
    # Print PCA results
    print("Explained variance first 10: ", pca_pip.named_steps.pca.explained_variance_ratio_[:10])
    print("All explained variance: ", pca_pip.named_steps.pca.explained_variance_ratio_.sum())
    print("Number of PCs: ", pca_pip.named_steps.pca.n_components_)
    
    # Transform test data
    pca_test = pca_pip.transform(data_features.loc[ind_val, cols])
    
    return pca_pip, pca_train, pca_test


def run_clustering(data_train, data_test, metadata, ind_train, ind_val, n_clusters, min_prob_accept, seed):
    """Run Gaussian Mixture Model clustering on data."""
    # Initialize result dataframes
    df_profile_clustering_train = metadata.loc[ind_train, ["Plate", "Well", "Treatment"]].reset_index(drop=True)
    df_profile_clustering_test = metadata.loc[ind_val, ["Plate", "Well", "Treatment"]].reset_index(drop=True)
    
    # Fit GMM
    gm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gm.fit(data_train)
    
    # Get probabilities and memberships for training data
    membership_train_proba = gm.predict_proba(data_train)
    membership_train = membership_train_proba.argmax(axis=1)
    
    # Accept clusters with probability above threshold
    membership_train_accept = membership_train_proba.max(axis=1) > min_prob_accept
    
    # Sort clusters by first PC value
    sorted_membership = np.argsort(gm.means_[:, 0]).tolist()
    
    # Save cluster information
    sorted_gm_means = gm.means_[np.array(sorted_membership), :]
    sorted_gm_covariances = gm.covariances_[np.array(sorted_membership), :]
    
    # Map original membership to sorted membership
    membership_train_sorted = [sorted_membership.index(mem) for mem in membership_train]
    
    # Assign membership to training data
    df_profile_clustering_train["membership"] = [
        str(member + 1) if membership_train_accept[i] else '-1' 
        for i, member in enumerate(membership_train_sorted)
    ]
    
    # Process test data
    membership_test_proba = gm.predict_proba(data_test)
    membership_test = membership_test_proba.argmax(axis=1)
    membership_test_accept = membership_test_proba.max(axis=1) > min_prob_accept
    membership_test_sorted = [sorted_membership.index(mem) for mem in membership_test]
    
    df_profile_clustering_test["membership"] = [
        str(member + 1) if membership_test_accept[i] else '-1' 
        for i, member in enumerate(membership_test_sorted)
    ]
    

    
    return df_profile_clustering_train, df_profile_clustering_test

def run_leiden_clustering(data_train,data_test,metadata,ind_train,ind_val,resolution_parameter,seed=42, distance_metric='cosine'):

    # Initialize result dataframes
    df_profile_clustering_train = metadata.loc[ind_train, ["Plate", "Well", "Treatment"]].reset_index(drop=True)
    df_profile_clustering_test = metadata.loc[ind_val, ["Plate", "Well", "Treatment"]].reset_index(drop=True)
    
    ## run clustering leiden on train and use labels for test
    membership_train, graph_train = train_leiden_clustering(data_train,n_neighbors=15,resolution_parameter = resolution_parameter,seed=seed, distance_metric=distance_metric)

    ## sorted memebership based on first PC
    mean_pc1 = np.array([data_train.loc[membership_train == i, "PC1"].mean() for i in range(len(set(membership_train)))])
    sorted_membership = np.argsort(mean_pc1).tolist()
    # Map original membership to sorted membership
    membership_train_sorted = [str(sorted_membership.index(mem) + 1) for mem in membership_train]

    df_profile_clustering_train["membership"] = membership_train_sorted
    
    ## KNN for test
    # Train KNN classifier using the training data and their Leiden cluster labels
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='cosine')
    knn.fit(data_train, membership_train_sorted)
    
    # Predict cluster labels for test data
    membership_test = knn.predict(data_test)
    
    # Assign the predicted membership to the test data
    df_profile_clustering_test["membership"] = membership_test
    
    return df_profile_clustering_train, df_profile_clustering_test
    

def train_leiden_clustering(data, n_neighbors=15, resolution_parameter=1.0, seed=42, distance_metric='cosine'):
    """
    Apply Leiden clustering to PCA data
    
    Parameters:
    -----------
    data : pandas DataFrame or numpy array
        The PCA data to cluster
    n_neighbors : int, default=15
        Number of neighbors to use when constructing the graph
    resolution_parameter : float, default=1.0
        Higher resolution means more clusters
    seed : int, default=42
        Random seed for reproducibility
    distance_metric : str, default='cosine'
        Metric to use for distance computation
        
    Returns:
    --------
    membership : numpy array
        Cluster assignments for each data point
    graph : igraph.Graph
        The graph used for clustering
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
        
    # Create k-nearest neighbors graph
    adjacency_matrix = kneighbors_graph(
        data_array, 
        n_neighbors=n_neighbors, 
        mode='distance', 
        metric=distance_metric,
        include_self=False
    )
    
    # Convert distances to similarities (using Gaussian kernel)
    adjacency_matrix.data = np.exp(-adjacency_matrix.data**2)
    
    # Make the graph symmetric (undirected)
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
    
    # Create igraph Graph from adjacency matrix
    sources, targets = adjacency_matrix.nonzero()
    weights = adjacency_matrix[sources, targets].A1
    
    edges = list(zip(sources.tolist(), targets.tolist()))
    graph = ig.Graph(edges=edges, directed=False)
    graph.es['weight'] = weights
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution_parameter,
        seed=seed
    )
    
    # Get cluster assignments
    membership = np.array(partition.membership)
    
    return membership, graph


#     ## plot memebership_aggregated for test
#     memebership_aggregated_plot(df_profile_clustering_test, train_figure_path, color_discrete_map)


def process_membership_aggregation(df_profile_clustering, nb_cluster):
    """Process membership data for aggregation."""

    list_sorted_cluster_cols = [f'cluster_{i+1}' for i in range(nb_cluster)]
    # Filter out rejected clusters
    agg_mem = df_profile_clustering.loc[df_profile_clustering.membership != '-1', :].copy()
    agg_mem.reset_index(drop=True, inplace=True)

    
    # Count occurrences of each membership per group
    agg_mem = agg_mem.groupby(['Plate', 'Well', 'Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})
    
    # Pivot to get columns for each cluster
    agg_mem = agg_mem.pivot_table(index=['Plate', 'Well', 'Treatment'], columns='membership', values='count', fill_value=0)
    
    # Normalize by row sums
    agg_mem = agg_mem.div(agg_mem.sum(axis=1), axis=0)
    
    # Rename columns to cluster_X format
    agg_mem.rename(columns=lambda x: f'cluster_{x}', inplace=True)
    
    # Add missing columns with zeros
    for col in list_sorted_cluster_cols:
        if col not in agg_mem.columns:
            agg_mem[col] = 0
    
    # Sort columns in order
    agg_mem = agg_mem[list_sorted_cluster_cols]
    
    # Reset index
    agg_mem.reset_index(inplace=True)
    
    return agg_mem

def calculate_similarities(profiles):
    """Calculate similarities between train and test profiles."""
    test_plates = list(profiles.keys())
    profile_names = list(profiles[test_plates[0]].keys())
    treatments = profiles[test_plates[0]][profile_names[0]]["train"].Treatment.unique()
    
    # Create dataframes for results
    indices = pd.MultiIndex.from_product([treatments, profile_names])
    df_sim_same_treatment = pd.DataFrame(index=indices, columns=test_plates)
    
    df_sim_activated_inactivated = pd.DataFrame(index=profile_names, columns=test_plates)
    
    # Calculate similarities
    for test_plate in test_plates:
        for profile_name in profile_names:
            tr = profiles[test_plate][profile_name]["train"].groupby("Treatment").mean(numeric_only=True)
            ts = profiles[test_plate][profile_name]["test"].groupby("Treatment").mean(numeric_only=True)
            
            # Same treatment similarities
            for treatment in treatments:
                df_sim_same_treatment.loc[(treatment, profile_name), test_plate] = 1 - cosine(tr.loc[treatment], ts.loc[treatment])
            
            # Activated vs inactivated in test set
            df_sim_activated_inactivated.loc[profile_name, test_plate] = 1 - cosine(ts.loc["LPS/Nigericin"], ts.loc["Unprimed/Unactivated"])
    
    # Calculate summary statistics
    a = df_sim_activated_inactivated.mean(axis=1).reset_index().rename(
        columns={0: "activated_inactivated_similarities", "index": "level_1"}
    )
    
    b = df_sim_same_treatment.reset_index().groupby("level_1")[test_plates].mean().mean(axis=1).reset_index().rename(
        columns={0: "same_treatment_similarities"}
    )
    
    data = pd.merge(a, b, on="level_1").rename(columns={"level_1": "Profile"})
    data["ratio"] = data["same_treatment_similarities"] / data["activated_inactivated_similarities"]
    
    return data

def run_classification(profiles):
    """Run different classifiers and calculate F1 scores."""
    classifiers = {
        "linear SVM": LinearSVC(random_state=42), 
        "Nearest Neighbors": KNeighborsClassifier(3), 
        "RBF SVM": SVC(random_state=42), 
        "LDA": LDA(), 
        "Naive Bayes": GaussianNB()
    }
    
    test_plates = list(profiles.keys())
    profile_names = list(profiles[test_plates[0]].keys())
    treatments = profiles[test_plates[0]][profile_names[0]]["train"].Treatment.unique()
    
    indices = pd.MultiIndex.from_product([classifiers.keys(), profile_names])
    df_classification_f1_score = pd.DataFrame(index=indices, columns=test_plates)
    
    for test_plate in test_plates:
        for profile_name in profile_names:
            tr = profiles[test_plate][profile_name]["train"]
            x_train = tr.select_dtypes(include='number')
            y_train = tr["Treatment"]
            
            ts = profiles[test_plate][profile_name]["test"]
            x_test = ts.select_dtypes(include='number')
            y_test = ts["Treatment"]
            
            
            for classifier_name, classifier in classifiers.items():
                if x_train.shape[1] < 2:
                    df_classification_f1_score.loc[(classifier_name, profile_name), test_plate] = np.nan
                    continue
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                df_classification_f1_score.loc[(classifier_name, profile_name), test_plate] = f1_score(
                    y_test, y_pred, average="macro"
                )
    
    # Format results for visualization
    data = df_classification_f1_score.mean(axis=1).reset_index().rename(
        columns={0: "F1 Score", "level_0": "Classifier", "level_1": "Profile"}
    )
    
    return data

def visualize_similarities(data, output_path):
    """Create visualization for similarities and save it."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data["Profile"], 
        y=data["activated_inactivated_similarities"], 
        name="activated_inactivated_similarities", 
        marker_color="blue"
    ))
    
    fig.add_trace(go.Bar(
        x=data["Profile"], 
        y=data["same_treatment_similarities"], 
        name="same_treatment_similarities", 
        marker_color="red"
    ))
    
    fig.add_trace(go.Bar(
        x=data["Profile"], 
        y=data["ratio"], 
        name="ratio", 
        marker_color="green"
    ))
    
    fig.update_layout(
        title="Treatment Similarity Metrics",
        barmode="group",
        xaxis_title="Profile",
        yaxis_title="Similarity Score",
        legend_title="Metric"
    )
    
    # Save the figure
    fig.write_html(os.path.join(output_path, "treatment_similarities.html"))
    
    return fig

def visualize_classification(data, output_path):
    """Create visualization for classification results and save it."""
    fig = px.bar(
        data, 
        x="Profile", 
        y="F1 Score", 
        color="Classifier", 
        barmode="group",
        title="Classification F1 Scores by Profile and Classifier"
    )
    
    # Save the figure
    fig.write_html(os.path.join(output_path, "classification_f1_scores.html"))
    
    return fig
