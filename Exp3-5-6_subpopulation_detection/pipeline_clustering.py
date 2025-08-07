#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import utils
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import plotly

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config_path):
    """Main function to run the pipeline profiling."""
    # Load configuration
    config = load_config(config_path)
    
    # Create directories
    output_file = config.get('output_file')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    paths = utils.create_directories(base_path)
    
    # Set paths
    params_path = paths['params_path']
    train_figure_path = paths['train_figure_path']
    base_path_csv = paths['base_path_csv']
    results_figure_path = paths["results_figure_path"]
    crops_visu_path = paths["crops_visu_path"]
    
    # Load data raise error if data_path is not specified
    data_path = config.get('data_path')
    if not data_path:
        raise ValueError("data_path must be specified in config")
    print(f"Loading data from: {data_path}")
    alldata = pd.read_feather(data_path)
    
    # Generate color map
    color_discrete_map = utils.generate_color_map()
    utils.save_color_map(color_discrete_map, params_path)
    
    # Extract parameters from config
    harmonize = config.get('harmonize', False)
    vars_use_harmony = config.get('vars_use_harmony', 'Plate')
    n_PCs_clustering = config.get('n_PCs_clustering', 10)
    feature_type = config.get('feature_type')
    channel_to_use = config.get('channel_to_use', ['all'])
    color_range = config.get('color_range', 1)
    seed = config.get('seed', 42)
    min_prob_accept = config.get('min_prob_accept', 0.7)
    nb_clusters = config.get('nb_clusters', [4, 25])
    chans_inorder = config.get('chans_inorder', [])
    numeric_cols_metadata = config.get('numeric_cols_metadata', [])
    test_size = config.get('test_size', 0.3)
    save_crops = config.get('save_crops', False)
    crop_image_width = config.get('crop_image_width', 140)
    staining = config.get('staining')

    # Save parameters
    params = {
        'harmonize': harmonize,
        'vars_use_harmony': vars_use_harmony,
        'n_PCs_clustering': n_PCs_clustering,
        'channel_to_use': channel_to_use,
        'color_range': color_range,
        'seed': seed,
        'min_prob_accept': min_prob_accept,
        'nb_clusters': nb_clusters,
        'plate_list': alldata.Plate.unique().tolist(),
        'test_size': test_size

    }
    
    with open(os.path.join(params_path, "params.json"), "w") as outfile:
        json.dump(params, outfile)
    
    
    ### seperate features by channel from deep data
    Fingerprints_to_use = utils.feature_sepration(alldata,feature_type,chans_inorder,numeric_cols_metadata)
    print(Fingerprints_to_use.keys())
        
    # Split features and metadata
    data_features = alldata[Fingerprints_to_use["all"]].copy().reset_index(drop=True)
    metadata = alldata.drop(Fingerprints_to_use["all"], axis=1)
    metadata.reset_index(drop=True, inplace=True)
    
    # Initialize profiles dictionary
    profiles = {}
    
    # Process each plate
    print("Starting the pipeline...")
    
    # Get columns for selected channels
    cols = []
    for chan in channel_to_use:
        cols += Fingerprints_to_use[chan]
    
    ###############
    ## train/test split
    ###############
    # Prepare train/test data
    random_state =  np.random.RandomState(seed=seed)
    ind_train,ind_val= train_test_split(metadata.index,stratify=metadata["Treatment"],random_state=random_state, test_size=test_size)
    
    ###############
    ## PCA
    ###############
    print("Running PCA...")
    pca_pip, pca_train, pca_test = utils.run_pca(data_features, ind_train, ind_val, cols, n_PCs_clustering, seed,train_figure_path,feature_type,Fingerprints_to_use)
    
    ###############
    ## harmonization
    ###############

    # Apply harmonization if requested
    if harmonize:
        print("Running harmony batch correction...")
        data_train_for_clustering, data_test_for_clustering = utils.apply_harmony(
            pca_train, pca_test, metadata, ind_train, ind_val, vars_use_harmony
        )

    else:
        print("Harmony batch correction set to False")
        # Train data
        data_train_for_clustering = pd.DataFrame(pca_train)
        PC_columns = [f'PC{i+1}' for i in range(pca_pip.named_steps.pca.n_components_)]
        data_train_for_clustering.columns = PC_columns
        
        # Test data
        data_test_for_clustering = pd.DataFrame(pca_test)
        data_test_for_clustering.columns = PC_columns
    
    ###############
    ## umap
    ###############
    print("Running UMAP...")
    ### umap on PCA
    visualizer, df_transformed_train, df_transformed_test = utils.run_umap(data_train_for_clustering, data_test_for_clustering, seed,metadata,ind_train,ind_val)
    
    ###############
    ## clustering
    ###############

    print("Running clustering analysis...")
    ## if nb_clusters is int skip it and else find optimal number of clusters
    if isinstance(nb_clusters, int):
        print(f"Clustering with {nb_clusters} clusters...")
        
    elif isinstance(nb_clusters, list):
        print("Finding optimal number of clusters...")
        ## TO DO: code find_optimal_clusters
        nb_clusters = utils.find_optimal_clusters(data_train_for_clustering, metadata,ind_train,seed,nb_clusters,train_figure_path)
        print(f"Optimal number of clusters: {nb_clusters}")
    else:
        raise ValueError("nb_clusters must be int or list")
        
    
    membership_train,membership_train_proba,membership_test,membership_test_proba,sorted_gm_means,sorted_gm_covariances = utils.run_clustering(
        data_train_for_clustering, data_test_for_clustering, 
        metadata, ind_train, ind_val, nb_clusters, min_prob_accept, seed,train_figure_path
    )
    df_transformed_train["membership"] = membership_train
    df_transformed_train["membership_prob"] = membership_train_proba
    df_transformed_test["membership"] = membership_test
    df_transformed_test["membership_prob"] = membership_test_proba

    ###############
    ## save results train figures
    ###############
    print("Saving clustering results...")
    
    ## plot umap with color memebership
    f1 = px.scatter(df_transformed_train, x = "emb1", y = "emb2",
        color = "membership", hover_data = ["Plate","Well","Treatment"] ,
        color_discrete_map = color_discrete_map,opacity=0.3, title= f"all clusters train set")
    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

    f1 = px.scatter(df_transformed_test, x = "emb1", y = "emb2",
            color = "membership", hover_data = ["Plate","Well","Treatment"] ,color_discrete_map = color_discrete_map,opacity=0.3, title= f"all clusters in test set")
    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

    ## similarity between clusters
    similarity_heatmap = utils.similarity_between_clusters(sorted_gm_means,train_figure_path)

    ## plot membership aggregated
    for plate in df_transformed_test.Plate.unique():
        df_grouped = df_transformed_test[df_transformed_test.Plate == plate].groupby(['Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})
        
        f1 = px.histogram(df_grouped,x = "Treatment", y = "count",color = "membership",barnorm = "percent", title = f"clusters per Treatment {plate}",color_discrete_map = color_discrete_map)
        f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))
    
    ## important features per cluster
    X_train_before_pca = data_features.loc[ind_train].reset_index(drop=True)
    X_train_before_pca["membership"] = membership_train
    utils.important_features_per_cluster(X_train_before_pca,train_figure_path,feature_type,Fingerprints_to_use)

    ###############
    ## save results final figures
    ###############
    
    ## plot importance of clusters for activated or non activated cells
    df_transformed_train, df_transformed_test, membership_importance_value_train_map, membership_importance_value_test_map = utils.assign_memebership_importance(df_transformed_train, df_transformed_test)
    membership_importance_value_scaled = {member:(value*0.5/color_range)+0.5 for member,value in membership_importance_value_test_map.items()}
    list_colors = plotly.colors.sample_colorscale("jet",membership_importance_value_scaled.values())
    membership_cmap = dict(zip(membership_importance_value_scaled.keys(), list_colors))

    f1 = px.scatter(df_transformed_test, x = "emb1", y = "emb2",opacity = 0.3,
                            color = "membership", hover_data = ["Plate","Well","Treatment","membership_importance_value"]
                            ,color_discrete_map=membership_cmap ,title= "culsters with importance values on DMSO")
    f1.write_html(os.path.join(results_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))
    
    ## membership aggregated all
    df_grouped = df_transformed_test.groupby(['Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})
    
    f1 = px.histogram(df_grouped,x = "Treatment", y = "count",color = "membership",barnorm = "percent", title = f"clusters per Treatment all",color_discrete_map = color_discrete_map)
    f1.write_html(os.path.join(results_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))
    ## df_grouped
    df_grouped.to_csv(os.path.join(base_path_csv,"clusters_per_treatment.csv"),index=False)

    ## save number of cell per cluster per treatment
    train_value_counts = df_transformed_train.membership.value_counts()
    test_value_counts = df_transformed_test.membership.value_counts()

    cluster_count_train = px.bar(x=train_value_counts.index, y =  train_value_counts.values, title= "number of samples per cluster train")
    cluster_count_train.write_image(os.path.join(results_figure_path,cluster_count_train.layout.__getattribute__("title")["text"] + ".png"))
    cluster_count_test = px.bar(x=test_value_counts.index, y =  test_value_counts.values, title= "number of samples per cluster test")
    cluster_count_test.write_image(os.path.join(results_figure_path,cluster_count_test.layout.__getattribute__("title")["text"] + ".png"))
    

    # Training data aggregation
    agg_mem_train = utils.process_membership_aggregation(
        df_transformed_train, nb_clusters
    )
        
    # Test data aggregation
    agg_mem_test = utils.process_membership_aggregation(
        df_transformed_test, nb_clusters
    )

    ## save some data

    ## save whole dataset results
    df_all_transformed = pd.concat([df_transformed_train,df_transformed_test],ignore_index=True)
    df_all_transformed.to_feather(os.path.join(base_path_csv,"umap_all_clusters"+".fth"))
    ## save train set results for crop visualization
    df_transformed_train.to_feather(os.path.join(base_path_csv,"umap_train_clusters"+".fth"))
    ## aggeregated clusters
    agg_mem_test.to_csv(os.path.join(base_path_csv,"aggregated_clusters_test.csv"))
    agg_mem_train.to_csv(os.path.join(base_path_csv,"aggregated_clusters_train.csv"))

    ###############
    ## create graphs and analysis clusters
    ###############
    utils.graph_clusters(sorted_gm_means,membership_importance_value_test_map,agg_mem_test,results_figure_path)

    ##############
    ## save crops
    ##############
    if save_crops:
        utils.save_crops(df_transformed_train,feature_type,staining,chans_inorder,crop_image_width,crops_visu_path)


    print(f"Pipeline completed. Results saved to {base_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
