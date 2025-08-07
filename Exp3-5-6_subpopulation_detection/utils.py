#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cosine
import umap
import plotly.express as px
import sys
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from kneed import KneeLocator
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_distances
import networkx as nx
from sklearn.feature_selection import f_classif
import operator
from pyvis.network import Network
from joblib import Parallel, delayed
from glob import glob
import pillow_jxl
from PIL import Image
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def feature_sepration(df,feature_type, chans,numeric_cols_metadata):
    if feature_type == "deep":
        return feature_seprations_deep(df, chans)
    elif feature_type == "non-deep":
        ## only numerics columns
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in features if col not in numeric_cols_metadata]
        return feature_seperations_nondeep(features, chans)
    else:
        raise ValueError("Invalid feature type. Must be 'deep' or 'non-deep'")

## features seperation
def feature_seperations_nondeep(features, chans):
    features_to_use = {chan:[] for chan in chans}
    features_to_use["others"] = []
    for col in features:
        key = "others"
        value_min = np.inf
        for chan in features_to_use.keys():
            val = col.find(chan)
            if val == -1:
                pass
            elif val < value_min:
                value_min =  val
                key = chan
        features_to_use[key].append(col)

    features_to_use["all"] =  features

    return features_to_use

def feature_seprations_deep(df,chans):

    Fingerprints_to_use = dict()
    for chan in chans:
        Fingerprints_to_use[chan] = df.columns[df.columns.str.contains(f"Feature_{chan}")].tolist()
    Fingerprints_to_use["all"] = []
    for chan in chans:
        Fingerprints_to_use["all"] += Fingerprints_to_use[chan]

    return Fingerprints_to_use

def create_directories(base_path):
    """Create necessary directories for saving outputs."""
    paths = {
        'params_path': os.path.join(base_path, 'params'),
        'train_figure_path': os.path.join(base_path, 'train_figures'),
        'base_path_csv': os.path.join(base_path, 'csv'),
        'results_figure_path' : os.path.join(base_path, 'results_figures'),
        "crops_visu_path": os.path.join(base_path,"crops_visualization")
        
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def generate_color_map():
    """Generate a color map for visualization."""
    color_discrete_map = {str(i):"rgb"+ str(tuple(np.random.choice(256,3).tolist())) for i in range(-1, 1000)}
    return color_discrete_map

def save_color_map(color_map, params_path):
    """Save the color map to a JSON file."""
    with open(os.path.join(params_path, "color_discrete_map.json"), "w") as f:
        json.dump(color_map, f)



def sillhoute_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    labels = estimator.predict(X)
    return silhouette_score(X,labels)
def run_pca(data_features, ind_train, ind_val, cols, n_pcs, seed, train_figure_path,feature_type,Fingerprints_to_use):
    """Run PCA on training and test data."""
    pca = PCA(n_components=n_pcs, random_state=seed)
    pca_pip = Pipeline(steps=[("pca", pca)])
    
    # Train PCA
    pca_train = pca_pip.fit_transform(data_features.loc[ind_train, cols])
    
    # Print PCA results
    print("Explained variance first 10: ", pca_pip.named_steps.pca.explained_variance_ratio_[:10])
    print("All explained variance: ", pca_pip.named_steps.pca.explained_variance_ratio_.sum())
    print("Number of PCs: ", pca_pip.named_steps.pca.n_components_)

    pca_weights = (pca_pip.named_steps.pca.components_.T)
    abs_pca_weights = np.abs(pca_weights)
    g = sns.clustermap(data = abs_pca_weights,yticklabels = pca_pip.feature_names_in_,
                        xticklabels = ['PC{}'.format(i + 1) for i in range(pca_pip.named_steps.pca.n_components_)],figsize=(10,200),col_cluster=False)
    g.figure.suptitle( "contribution of each feature in PCs space")
    g.figure.savefig(os.path.join(train_figure_path,f"contribution of each feature in PCs space.png"))
    plt.close(g.figure)

    if feature_type == "deep":
        ## aggeregate pca by channels
        channs = list(Fingerprints_to_use.keys())
        channs.remove("all")
        agg_pca_weights = np.zeros((len(channs),pca_pip.named_steps.pca.n_components_))
        for ind,key in enumerate(channs):
            indices = np.where(np.in1d(pca_pip.feature_names_in_, np.array(Fingerprints_to_use[key])))[0]
            agg_pca_weights[ind] = abs_pca_weights[indices].mean(axis=0)
        
        g = sns.clustermap(data = agg_pca_weights,yticklabels = channs,
                            xticklabels = ['PC{}'.format(i + 1) for i in range(pca_pip.named_steps.pca.n_components_)],figsize=(10,10),col_cluster=False)
        g.figure.suptitle( "aggregated per channel contribution of each feature in PCs space")
        g.figure.savefig(os.path.join(train_figure_path,"aggregated per channel contribution of each feature in PCs space.png"))
        plt.close(g.figure)
    
    # save loadings 
    loading_all = pca_weights * np.sqrt(pca_pip.named_steps.pca.explained_variance_)
    loadings = pd.DataFrame(
    data=np.linalg.norm(loading_all,ord=2,axis=1)**2, 
    columns=["var_explained"],
    index=pca_pip.feature_names_in_
    )
    loadings.sort_values(by="var_explained",ascending=False,inplace=True)
    loadings.reset_index(inplace=True)
    loadings.rename({"index" : "Features"},axis = 1,inplace=True)
    f1 = px.bar(loadings, x = "Features",y="var_explained",title="Features' explained variance in PCs space")
    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))
    # Transform test data
    pca_test = pca_pip.transform(data_features.loc[ind_val, cols])
    
    return pca_pip, pca_train, pca_test

def run_umap(data_train, data_test, seed,metadata,ind_train,ind_val):
    """Run UMAP on training and test data."""
    visualizer = umap.UMAP(n_components=2,n_neighbors = 100,min_dist=0.1,random_state=seed) 
    
    # Train UMAP
    df_umap_train = pd.DataFrame(visualizer.fit_transform(data_train), columns = ["emb1","emb2"])
    df_umap_train = pd.concat([df_umap_train,metadata.loc[ind_train].reset_index(drop=True)],axis=1)
    
    # Transform test data
    df_umap_test = pd.DataFrame(visualizer.transform(data_test), columns = ["emb1","emb2"])

    df_umap_test = pd.concat([df_umap_test,metadata.loc[ind_val].reset_index(drop=True)],axis=1)
    
    return visualizer, df_umap_train, df_umap_test

def apply_harmony(pca_train, pca_test, metadata, ind_train, ind_val, vars_use_harmony):
    """Apply Harmony batch correction if enabled."""
    try:
        import harmonypy as hm
        
        # Harmonize train
        ho_train = hm.run_harmony(pca_train, metadata.loc[ind_train].reset_index(drop=True), 
                                  vars_use_harmony, max_iter_harmony=20)
        
        # Harmonize test
        ho_test = hm.run_harmony(pca_test, metadata.loc[ind_val].reset_index(drop=True), 
                                vars_use_harmony, max_iter_harmony=20)
        
        # Create dataset harmonized - train
        data_train_for_clustering = pd.DataFrame(ho_train.Z_corr.T)
        PC_columns = [f'PC{i+1}' for i in range(ho_train.Z_corr.shape[0])]
        data_train_for_clustering.columns = PC_columns
        
        # Create dataset harmonized - test
        data_test_for_clustering = pd.DataFrame(ho_test.Z_corr.T)
        data_test_for_clustering.columns = PC_columns
        
        return data_train_for_clustering, data_test_for_clustering
    
    except ImportError:
        print("harmonypy module not found. Skipping harmonization.")
        return None, None



def run_clustering(data_train, data_test, metadata, ind_train, ind_val, n_clusters, min_prob_accept, seed,train_figure_path):
    """Run Gaussian Mixture Model clustering on data."""
    
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

    np.save(os.path.join(train_figure_path,"GMM_clusters_means.npy"),sorted_gm_means)
    np.save(os.path.join(train_figure_path,"GMM_clusters_covariances.npy"),sorted_gm_covariances)
    
    # Map original membership to sorted membership
    membership_train_sorted = [sorted_membership.index(mem) for mem in membership_train]
    
    # Assign membership to training data
    membership_train = [
        str(member + 1) if membership_train_accept[i] else '-1' 
        for i, member in enumerate(membership_train_sorted)
    ]
    
    # Process test data
    membership_test_proba = gm.predict_proba(data_test)
    membership_test = membership_test_proba.argmax(axis=1)
    membership_test_accept = membership_test_proba.max(axis=1) > min_prob_accept
    membership_test_sorted = [sorted_membership.index(mem) for mem in membership_test]
    
    membership_test = [
        str(member + 1) if membership_test_accept[i] else '-1' 
        for i, member in enumerate(membership_test_sorted)
    ]
    

    
    return membership_train,membership_train_proba.max(axis=1), membership_test,membership_test_proba.max(axis=1),sorted_gm_means,sorted_gm_covariances

def assign_memebership_importance(df_profile_clustering_train, df_profile_clustering_test):
    """Assign importance value to each membership."""
    def value_membership(member,df):
        inactive_ind = (df.Treatment == "Unprimed/Unactivated")
        active_ind = (df.Treatment == "LPS/Nigericin")
        member_nb = ((df.membership == member) & (active_ind | inactive_ind)).sum()
        nb_member_active = ((df.membership == member) & active_ind).sum()
        nb_member_inactive = ((df.membership == member) & inactive_ind).sum()
        return (nb_member_active - nb_member_inactive) / member_nb
    
    membership_importance_value_train = {member:value_membership(member,df_profile_clustering_train) for member in df_profile_clustering_train.membership.unique()}
    membership_importance_value_test = {member:value_membership(member,df_profile_clustering_test) for member in df_profile_clustering_test.membership.unique()}
    df_profile_clustering_train["membership_importance_value"] = df_profile_clustering_train.membership.map(membership_importance_value_train)
    df_profile_clustering_test["membership_importance_value"] = df_profile_clustering_test.membership.map(membership_importance_value_test)
    
    return df_profile_clustering_train, df_profile_clustering_test, membership_importance_value_train, membership_importance_value_test

def similarity_between_clusters(sorted_gm_means,train_figure_path):
    distances_mat = distance_matrix(sorted_gm_means,sorted_gm_means,p=2)
    distances_mat = distances_mat / distances_mat.max()
    similarity_heatmap = pd.DataFrame(1-distances_mat, index = list(map(str,list(range(1,sorted_gm_means.shape[0]+1)))), columns = list(map(str,list(range(1,sorted_gm_means.shape[0]+1)))))
    g = sns.clustermap(similarity_heatmap)
    g.figure.suptitle( "Similarity between clusters")
    g.figure.savefig(os.path.join(train_figure_path,"Similarity_between_clusters.png"))
    return similarity_heatmap

def important_features_per_cluster(X_train_before_pca,train_figure_path,feature_type,Fingerprints_to_use):


    list_labels = X_train_before_pca.membership.unique().tolist()
    cols_feature = list(X_train_before_pca.columns)
    cols_feature.remove("membership")
    cols_feature_array = np.array(cols_feature)
    features_important = dict()

    f_stats_features = np.zeros((len(list_labels),len(cols_feature)))
    for ii,label in enumerate(list_labels):
        y = (X_train_before_pca.membership == label).astype(int)
        f_stats,p_values = f_classif(X_train_before_pca[cols_feature],y=y)
        ind_sorts = np.argsort(f_stats)[::-1]
        features_top = cols_feature_array[ind_sorts[:10]]
        features_important[label] = list(features_top)
        f_stats_features[ii,:] = (f_stats - f_stats.min()) / (f_stats.max() - f_stats.min())

    g = sns.clustermap(data = f_stats_features.T,yticklabels = cols_feature,
                        xticklabels = list_labels ,figsize=(10,200),col_cluster=False)
    g.figure.suptitle( "feature importance for each cluster based on f-statistics")
    g.figure.savefig(os.path.join(train_figure_path,"feature importance for each cluster based on f-statistics.png"))

    ## save list of 10 important features per cluster
    with open(os.path.join(train_figure_path,"top_features_per_cluster.json"), 'w', encoding='utf-8') as f:
        json.dump(features_important, f, ensure_ascii=False, indent=4)

    if feature_type == "deep":
        ## aggeregate f_stats by channels
        channs = list(Fingerprints_to_use.keys())
        channs.remove("all")
        f_stats_agg = np.zeros((len(channs),len(list_labels)))
        for ii,key in enumerate(channs):
            indices = np.where(np.in1d(np.array(cols_feature), np.array(Fingerprints_to_use[key])))[0]
            f_stats_agg[ii] = f_stats_features[:,indices].mean(axis=1)

        g = sns.clustermap(data = f_stats_agg,yticklabels = channs,
                            xticklabels = list_labels ,figsize=(10,10),col_cluster=False)
        g.figure.suptitle("aggregated per channel feature importance for each cluster based on f-statistics")
        g.figure.savefig(os.path.join(train_figure_path,"aggregated per channel feature importance for each cluster based on f-statistics.png"))

def process_membership_aggregation(df_profile_clustering, nb_cluster):
    """Process membership data for aggregation."""

    list_sorted_cluster_cols = [f'cluster_{i+1}' for i in range(nb_cluster)]
    # Filter out rejected clusters
    agg_mem = df_profile_clustering.loc[df_profile_clustering.membership != '-1', :].copy()
    agg_mem.reset_index(drop=True, inplace=True)

    
    # Count occurrences of each membership per group
    agg_mem = agg_mem.groupby(['Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})
    
    # Pivot to get columns for each cluster
    agg_mem = agg_mem.pivot_table(index='Treatment', columns='membership', values='count', fill_value=0)
    
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

def graph_clusters(sorted_gm_means,membership_importance_value_test,agg_mem_Cntrl,final_figure_path):
    cluster_unactivated = int(min(membership_importance_value_test.items(), key=operator.itemgetter(1))[0])
    print("cluster_unactivated  ",cluster_unactivated)
    cluster_activated = int(max(membership_importance_value_test.items(), key=operator.itemgetter(1))[0])
    print("cluster_activated  ",cluster_activated)


    ### spaning tree visualization and top drug effects on it

    # distances_mat = scipy.spatial.distance_matrix(sorted_gm_means,sorted_gm_means)
    distances_mat = cosine_distances(sorted_gm_means,sorted_gm_means)
    distances_df = pd.DataFrame(distances_mat, index = list(map(str,list(range(1,sorted_gm_means.shape[0]+1)))), columns = list(map(str,list(range(1,sorted_gm_means.shape[0]+1)))))
    mask_keep = np.triu(np.ones(distances_df.shape), k=1).astype('bool').reshape(distances_df.size)
    sr = distances_df.stack()[mask_keep]
    df_edges_distances = sr.reset_index().rename({"level_0" : "cluster_src", "level_1": "cluster_des", 0 :  "distance"},axis=1 )
    df_edges_distances.to_csv(os.path.join(final_figure_path,"edges_distances.csv"))

    ## create graph
    G = nx.Graph()
    for i in df_edges_distances.index:
        G.add_edge(df_edges_distances.loc[i,"cluster_src"], df_edges_distances.loc[i,"cluster_des"], weight = (df_edges_distances.loc[i,"distance"]))

    ## minimum spaning tree
    T = nx.minimum_spanning_tree(G)



    # shortest path length from unactivated
    length=nx.single_source_shortest_path_length(T,str(cluster_unactivated))
    print("length",length)

    # add levels to T 
    attrs = {node:{"level" : length[node]} for node in T.nodes}
    nx.set_node_attributes(T, attrs)

    for i in agg_mem_Cntrl.index:
        # Create a PyVis Network
        net = Network(height='800px', width='100%', notebook=False, directed=False, layout=True)
        
        # Configure network options for hierarchical layout to prevent crossing lines
        net.set_options('''
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "shakeTowards": "roots",
                    "levelSeparation": 150,
                    "nodeSpacing": 200
                }
            },
            "nodes": {
                "font": {
                    "size": 16,
                    "face": "arial",
                    "color": "#000000",
                    "vadjust": -50
                },
                "shape": "circle",
                "size": 30,
                "borderWidth": 2,
                "shadow": true
            },
            "edges": {
                "font": {
                    "size": 14,
                    "face": "arial"
                },
                "width": 2,
                "color": {
                    "color": "#000000"
                }
            },
            "physics": {
                "enabled": false
            }
        }
        ''')
        
        # Add nodes with color attributes from the original code
        node_colors = agg_mem_Cntrl.loc[i,list(map(lambda x: f"cluster_{x}",list(T.nodes)))].values
        color_range = 0.15
        # Map node colors to a colormap similar to 'jet' (0 to 0.2 range)
        for node_idx, node in enumerate(T.nodes()):
            # Get the color value for this node and scale it to 0-0.2 range
            color_val = float(node_colors[node_idx])
            # Map color_val from 0-0.2 to a hex color in jet colormap
            if color_val > color_range:
                color_val = color_range
            # Scale to 0-1 for color mapping
            color_normalized = color_val / color_range
            
            # Simple jet-like colormap implementation: blue (0) to red (1)
            if color_normalized < 0.25:
                # Blue to cyan
                r = 0
                g = int(255 * color_normalized * 4)
                b = 255
            elif color_normalized < 0.5:
                # Cyan to green
                r = 0
                g = 255
                b = int(255 * (1 - (color_normalized - 0.25) * 4))
            elif color_normalized < 0.75:
                # Green to yellow
                r = int(255 * (color_normalized - 0.5) * 4)
                g = 255
                b = 0
            else:
                # Yellow to red
                r = 255
                g = int(255 * (1 - (color_normalized - 0.75) * 4))
                b = 0
            
            # Convert RGB to hex
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Add node with level for hierarchical layout and larger internal label
            net.add_node(node, label=node, level=T.nodes[node]['level'], color=hex_color, title=f"Cluster {node}: {color_val:.3f}")
        
        # Add edges with weight attributes - all black, same thickness, with weight labels
        for edge in T.edges(data=True):
            weight = edge[2]['weight']
            # All edges will be uniform black with the same thickness (set in options)
            # Add weight as a label on the edge
            net.add_edge(edge[0], edge[1], label=f"{weight:.2f}", title=f"Distance: {weight:.2f}")   
        tag = agg_mem_Cntrl.loc[i,"Treatment"]
        # Create HTML file name
        tag = tag.replace("/","-")
        html_path = os.path.join(final_figure_path, f"{tag}_Graph.html")
        
        # Save as interactive HTML file
        net.save_graph(html_path)

def save_crops(df_umap_train,feature_type,staining,chans_inorder,crop_image_width,crops_visu_path):
    df = df_umap_train[df_umap_train.membership_prob > 0.7]
    if feature_type == "deep":
        df["X"] = df.BBox_Left + (df.BBox_Width // 2)
        df["Y"] = df.BBox_Top + (df.BBox_Width // 2)
    
    if staining == "CP":
        chans_inorder = chans_inorder + ["BF"]
        


    Crop_Visualization_autosave(df,base_image_paths="/mnt/shares/O/BTSData/MeasurementData/MIG",base_path_save = crops_visu_path, nb_samples = 10,
                                        number_feild=9, BBox_width = crop_image_width,channels_names=chans_inorder)



def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)
def gmm_aic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the AIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.aic(X)

def find_optimal_clusters(data_train_for_clustering,metadata,ind_train,seed,nb_clusters,train_figure_path):

    PC_columns = data_train_for_clustering.columns
    data_train_for_clustering_metadata = pd.concat([data_train_for_clustering,metadata.loc[ind_train].reset_index(drop=True)],axis=1)
    
    ## grid search gmm params
    print("grid search gmm params")
    
    param_grid = { "n_components": range(nb_clusters[0],nb_clusters[1]+1), "tol": [1e-3,1e-2,1e-1]}
    grid_search = GridSearchCV(
    GaussianMixture(random_state=seed,n_init=1), param_grid=param_grid, scoring=dict(bic = gmm_bic_score, aic = gmm_aic_score, sillhoute = sillhoute_score), n_jobs=-1, refit= False, cv=3 #bic = gmm_bic_score, 
    )
    grid_search.fit(data_train_for_clustering_metadata[PC_columns])
    df = pd.DataFrame(grid_search.cv_results_)[["param_n_components","param_tol", "mean_test_bic","mean_test_aic","mean_test_sillhoute"]]
    df["mean_test_bic"] = np.log10(-df["mean_test_bic"])
    df["mean_test_aic"] = np.log10(-df["mean_test_aic"])
    name_file = "gmm_gridResults"
    df.to_csv(os.path.join(train_figure_path,name_file + ".csv"),index=False)

    ## use knee to find best number of clusters
    ## plot
    fig = px.line(df,x = "param_n_components",y = "mean_test_bic",color = "param_tol", title = f"finding optimal number of clusters")
    fig.add_vline(x=df.param_n_components[df.mean_test_bic.idxmin()], line_dash="dash", line_color="green")
    fig.write_image(os.path.join(train_figure_path, f"gmm_gridResults_bic" + ".png"))
    fig = px.line(df,x = "param_n_components",y = "mean_test_aic",color = "param_tol", title = f"finding optimal number of clusters")
    fig.add_vline(x=df.param_n_components[df.mean_test_aic.idxmin()], line_dash="dash", line_color="green")
    fig.write_image(os.path.join(train_figure_path, f"gmm_gridResults_aic" + ".png"))
    
    
    
    nb_clusters = df.param_n_components[df.mean_test_aic.idxmin()]
    return nb_clusters

def Crop_Visualization_autosave(df_transformed,base_image_paths,base_path_save, cluster_column = "membership", number_feild=25, feild_size = (2560,2160),
                                BBox_width = None, max_value_chans = None, channels_names = ["Nuc","NFKB","Cd45","ASC"],
                                nb_samples = 10, columns_table = ["Plate","Well","tags","membership"], sampling = "highest_prob", sort_column = "membership_prob"):
    """This function takes dataframe and arguments to create a figure from actual crops corresponding to each cluster

    Args:
        df_transformed (pd.Dataframe):
            dataframe refrence that figure has been generated with

        base_image_paths (str): the path to the latest folder that all the images exist in its subdirectories seperated by plate name folders

        base_path_save (str): the path to save the figure

        number_feild (int): number of fields in acqisition. Defaults to 25.

        feild_size (tuple, optional): size of each feild in pixels. Defaults to (2560,2160).

        BBox_width (int, optional): size of crops if not in the dataframe. Defaults to None.
        
        max_value_chans (list, optional): max value of each channel for visualization of crops in order. Defaults to None. ex. [1000,1000,1000,1000]

        channels_names (list, optional): channel names in order. Defaults to ["Nuc","NFKB","Cd45","ASC"].

        nb_samples (int, optional): maximum number of samples to show after box selection. Defaults to 10.

        columns_table (list, optional): columns to show in table of contents. consider change the width of table columns. Defaults to ["Plate","Well","tags","membership"].
        
        sampling (str, optional): sampling method to select the samples. Defaults to "highest_prob". options are "random" and "highest_prob"

        sort_column (str, optional): column to sort the dataframe before sampling. Defaults to "membership_prob".
    """

    if sampling == "highest_prob":
        df_transformed = df_transformed.groupby(cluster_column).apply(lambda x: x.sort_values(sort_column,ascending=False).head(2*nb_samples)).reset_index(drop=True)
    elif sampling == "random":
        df_transformed = df_transformed.groupby(cluster_column).sample(nb_samples*2).reset_index(drop=True)
    else:
        raise ValueError("sampling should be either 'highest_prob' or 'random'")
    
    
    df_transformed = find_feild(df_transformed,feild_size,number_feild,BBox_width)
    for cluster, sdf in df_transformed.groupby(cluster_column):
        nb_samples_cluster = min(nb_samples,len(sdf))

        if sampling == "highest_prob":
            sdf = sdf.sort_values(sort_column,ascending=False).head(nb_samples_cluster).reset_index(drop=True)
        elif sampling == "random":
            sdf = sdf.sample(nb_samples_cluster,ignore_index= True)
        

        crops_selected = load_crop_images(sdf,base_image_paths,len(channels_names),max_value_chans)
        f = go.Figure(make_subplots(nb_samples_cluster,len(channels_names),subplot_titles=[channels_names[i] for i in range(len(channels_names))]))
        f.add_traces(px.imshow(crops_selected.reshape((-1,crops_selected.shape[2],crops_selected.shape[3])), binary_string=True, facet_col=0,
                       facet_col_wrap=len(channels_names),zmin=0,zmax=65535).data)
        f.update_layout(height = nb_samples_cluster*200)
        f.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        # for annot in f.layout.annotations:
        #     print(annot)
        #     annot.update(visible = False)
        
        f.add_annotation(xref = f'x domain',
                yref= f"y domain",
                showarrow=False,
                font={'size' : 16},
                xanchor = "right",
                yanchor = "middle",
                x = -0.2,
                y = 0.5,
                text = f"prob:{sdf.loc[0,'membership_prob']:0.2f}",
                name = "0")
        for s in range(1,nb_samples_cluster):
            fig_number = len(channels_names)*s + 1
            f.add_annotation(xref = f'x{fig_number} domain',
                            yref= f"y{fig_number} domain",
                            showarrow=False,
                            font={'size' : 16},
                            xanchor = "right",
                            yanchor = "middle",
                            x = -0.2,
                            y = 0.5,
                            text = f"prob:{sdf.loc[s,'membership_prob']:0.2f}",
                            name = f'{s}'
                            )
        
        f.write_image(os.path.join(base_path_save,f"{cluster}.png"))
        ## table
        t = go.Figure([go.Table(
                            header=dict(values=["sample_number"] + columns_table,
                                        fill = dict(color='#C2D4FF')),
                            cells=dict(values=[sdf.index.__add__(1)] + [sdf[col] for col in columns_table],
                                    fill = dict(color='#F5F8FF')),
                            columnwidth = [0.1,0.25,0.05,0.5,0.1])])
        t.update_layout(width=400*len(columns_table))
        t.write_image(os.path.join(base_path_save,f"table_{cluster}.png"))

def find_feild(df,feild_size,number_feild,BBox_width = None):
    if BBox_width:
        df["BBox_Width"] = BBox_width
    
    df["feild"] = df[["X","Y"]].apply(lambda x: (int(x["X"] / feild_size[0]) + 1) + np.sqrt(number_feild) * (int(x["Y"] / feild_size[1])), axis=1)
    df[["X_f", "Y_f"]] = df[["X","Y"]].apply(lambda x: [x["X"] % feild_size[0] ,x["Y"] % feild_size[1]], axis=1,result_type='expand')

    ## exclude border images
    mask = (df.X_f + (df.BBox_Width // 2) < feild_size[0]) & (df.Y_f + (df.BBox_Width // 2) < feild_size[1]) & (df.X_f - (df.BBox_Width // 2) > 0) & (df.Y_f - (df.BBox_Width // 2) > 0)
    return df[mask].reset_index(drop=True)
    
def load_image(image_path,chan,max_value_chans):
    if image_path.endswith(".tif"):
        image_source = cv2.imread(image_path,-1)
    elif image_path.endswith(".jxl"):
        image_source = np.array(Image.open(image_path))
    else:
        raise ValueError("image path should end with .tif or .jxl")
    if max_value_chans:
        image_source[image_source > max_value_chans[chan]] = max_value_chans[chan]
    return image_source

def load_crop_images(df,base_image_paths,n_chans,max_value_chans = None,n_jobs = 10, Normalize= True):
    
    ## create list of images to load
    all_file_names = []
    chans_col = [f"ch{j}" for j in range(n_chans)]
    df[chans_col] = " "
    dict_chans = dict()
    for i in df.index:
        plate = df.loc[i,"Plate"]
        time = 1
        field = int(df.loc[i,"feild"])
        well = df.loc[i,"Well"]
        image_plate_path = glob(os.path.join(base_image_paths,f"{plate}*"))
        assert(len(image_plate_path) == 1)
        image_plate_path = image_plate_path[0]
        tif_files = glob(os.path.join(image_plate_path,"**",f"{plate}_{well}_T{time:04d}F{field:03d}*.tif"), recursive=True)
        jxl_files = glob(os.path.join(image_plate_path, "**", f"{plate}_{well}_T{time:04d}F{field:03d}*.jxl"), recursive=True)
        file_names = tif_files + jxl_files
        assert(len(file_names) == n_chans)
        file_names.sort()
        df.loc[i,chans_col] = file_names
        all_file_names.extend(file_names)
        dict_chans.update(zip(file_names,list(range(len(file_names)))))
    ## find unique files
    all_file_names = list(set(all_file_names))
    ## assigne index to df for channel
    df[chans_col] = df[chans_col].map(lambda x: all_file_names.index(x))
    ## load images 
    list_images = Parallel(n_jobs=n_jobs)(delayed(load_image)(file,dict_chans[file],max_value_chans) for file in all_file_names)
    list_images = np.stack(list_images,axis=0)

    ## crop images
    images_to_show = []
    for i in df.index:
        width = int(df.loc[i,"BBox_Width"])
        left = int(df.loc[i,"X_f"]) - (width // 2)
        top = int(df.loc[i,"Y_f"]) - (width // 2)
        inds_chans = df.loc[i,chans_col].values.astype(int)
        image_crop = list_images[inds_chans,top:top+width,left:left+width]
        images_to_show.append(image_crop)
    
    # ## go through images and pad zero to make them same size
    # max_width = max([image.shape[1] for image in images_to_show])
    # max_height = max([image.shape[2] for image in images_to_show])
    # ## pad zero before stacking
    # for i in range(len(images_to_show)):
    #     images_to_show[i] = cv2.copyMakeBorder(images_to_show[i], 0, max_height - images_to_show[i].shape[0], 0, max_width - images_to_show[i].shape[1], cv2.BORDER_CONSTANT, value=0)
    images_to_show = np.stack(images_to_show,axis=0)

    if Normalize:
        mins = images_to_show.min(axis=(2,3))[:,:,np.newaxis,np.newaxis]
        maxs = images_to_show.max(axis=(2,3))[:,:,np.newaxis,np.newaxis]

        images_to_show = (images_to_show - mins) / (maxs - mins)
        images_to_show =  images_to_show * 65535

    return images_to_show