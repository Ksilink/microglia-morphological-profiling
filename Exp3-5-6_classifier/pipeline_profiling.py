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
    train_figure_path = paths['QC_figure_path']
    base_path_csv = paths['base_path_csv']
    results_figure_path = paths["final_figure_path"]
    
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
    feature_type = config.get('feature_type', 'deep')
    channel_to_use = config.get('channel_to_use', ['all'])
    color_range = config.get('color_range', 1)
    seed = config.get('seed', 42)
    chans_inorder = config.get('chans_inorder', [])
    numeric_cols_metadata = config.get('numeric_cols_metadata', [])
    
    # Extract clustering parameters from config
    clustering_params = config.get('clustering_params', {})
    clustering_method = clustering_params.get('clustering_method', 'GMM')  # Default to GMM
    
    if clustering_method == 'GMM':
        min_prob_accept = clustering_params.get('min_prob_accept', 0.7)
        range_number_clusters = clustering_params.get('range_number_clusters', [4, 25])
        # Calculate steps - always have 20 steps
        cluster_values = list(range(range_number_clusters[0], range_number_clusters[1] + 1))

    elif clustering_method == 'leiden':
        distance_metric = clustering_params.get('distance_metric', 'cosine')
        range_resolution = clustering_params.get('range_resolution', [0.1, 5])
        # Calculate 20 evenly spaced resolution values
        resolution_step = (range_resolution[1] - range_resolution[0]) / 19  # 19 steps for 20 values
        resolution_values = [range_resolution[0] + i * resolution_step for i in range(20)]
    # Save parameters
    params = {
        'harmonize': harmonize,
        'vars_use_harmony': vars_use_harmony,
        'n_PCs_clustering': n_PCs_clustering,
        'channel_to_use': channel_to_use,
        'color_range': color_range,
        'seed': seed,
        'clustering_method': clustering_method,
        'plate_list': alldata.Plate.unique().tolist()
    }
    
    if clustering_method == 'GMM':
        params['min_prob_accept'] = min_prob_accept
        params['range_number_clusters'] = range_number_clusters
    else:  # leiden
        params['distance_metric'] = distance_metric
        params['range_resolution'] = range_resolution
    
    with open(os.path.join(params_path, "params.json"), "w") as outfile:
        json.dump(params, outfile)
    
    
    ### seperate features by channel from deep data
    Fingerprints_to_use = utils.feature_sepration(alldata,feature_type,chans_inorder,numeric_cols_metadata)
    print(Fingerprints_to_use.keys())
    if "others" in Fingerprints_to_use:
        print(Fingerprints_to_use["others"])
        
    # Split features and metadata
    data_features = alldata[Fingerprints_to_use["all"]].copy().reset_index(drop=True)
    metadata = alldata.drop(Fingerprints_to_use["all"], axis=1)
    metadata.reset_index(drop=True, inplace=True)
    
    # Initialize profiles dictionary
    profiles = {}
    
    # Process each plate
    print("Starting plate processing...")
    for plate in metadata.Plate.unique():
        print(f"Processing plate: {plate}")
        
        # Get columns for selected channels
        cols = []
        for chan in channel_to_use:
            cols += Fingerprints_to_use[chan]
        
        # Prepare train/test data
        ind_train = metadata[metadata.Plate != plate].index.tolist()
        ind_val = metadata[metadata.Plate == plate].index.tolist()
        
        # Run PCA
        print("Running PCA...")
        pca_pip, pca_train, pca_test = utils.run_pca(data_features, ind_train, ind_val, cols, n_PCs_clustering, seed)
        
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
        
        ### umap on PCA
        if "Exp06" in plate:
            print("Running UMAP...")
            visualizer, df_umap_train, df_umap_test = utils.run_umap(data_train_for_clustering, data_test_for_clustering, seed,metadata,ind_train,ind_val)
        
        # Aggregate PC profiles
        print("Aggregating PC profiles...")
        aggregated_train_pca_well = pd.concat(
            [data_train_for_clustering, metadata.loc[ind_train].reset_index(drop=True)], axis=1
        ).groupby(["Plate", "Well", "Treatment"], as_index=False)[PC_columns].mean().reset_index(drop=True)

        
        aggregated_test_pca_well = pd.concat(
            [data_test_for_clustering, metadata.loc[ind_val].reset_index(drop=True)], axis=1
        ).groupby(["Plate", "Well", "Treatment"], as_index=False)[PC_columns].mean().reset_index(drop=True)
        
        profiles_plate = {"pca": {"train": aggregated_train_pca_well, "test": aggregated_test_pca_well}}
        
        # Run clustering
        print("Running clustering analysis...")
        if clustering_method == 'GMM':
            for nb_cluster in cluster_values:
                print(f"  Clustering with {nb_cluster} clusters (GMM)...")
                df_profile_clustering_train, df_profile_clustering_test = utils.run_clustering(
                    data_train_for_clustering, data_test_for_clustering, 
                    metadata, ind_train, ind_val, nb_cluster, min_prob_accept, seed
                )
                
                ## ploting only for one plate
                if "Exp06" in plate:
                    ## plot umap with color memebership
                    df_umap_train["membership"] = df_profile_clustering_train["membership"]
                    df_umap_test["membership"] = df_profile_clustering_test["membership"]
                    f1 = px.scatter(df_umap_test, x = "emb1", y = "emb2",
                        color = "membership", hover_data = ["Plate","Well","Treatment"] ,color_discrete_map = color_discrete_map,opacity=0.3, title= f"umap_test_{plate}_clusters_{nb_cluster}")
                    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

                    ## plot membership aggregated
                    df_grouped = df_profile_clustering_test.groupby(['Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})

                    
                    df_grouped = df_grouped.pivot_table(index='Treatment', columns='membership', values='count', fill_value=0)
                    
                    # Normalize by row sums
                    df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

                    df_grouped = df_grouped.stack().reset_index().rename(columns={"level_1": "membership", 0: "percentage"})

                    f1 = px.bar(df_grouped,x = "Treatment", y = "percentage",color = "membership", title = f"agg_Treatment_{plate}_clusters_{nb_cluster}",color_discrete_map = color_discrete_map)
                    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

                    df_grouped.to_csv(os.path.join(base_path_csv, f"agg_Treatment_{plate}_clusters_{nb_cluster}.csv"))
                
                # Training data aggregation
                agg_mem_train = utils.process_membership_aggregation(
                    df_profile_clustering_train, nb_cluster
                )
                    
                # Test data aggregation
                agg_mem_test = utils.process_membership_aggregation(
                    df_profile_clustering_test, nb_cluster
                )
                
                profiles_plate[f"clustering_{nb_cluster}"] = {"train": agg_mem_train, "test": agg_mem_test}
        else:  # leiden
            for resolution in resolution_values:
                print(f"  Clustering with resolution {resolution:.2f} (Leiden)...")
                df_profile_clustering_train, df_profile_clustering_test = utils.run_leiden_clustering(
                    data_train_for_clustering, data_test_for_clustering,
                    metadata, ind_train, ind_val, resolution, seed, distance_metric
                )
                
                # Get number of clusters (needed for aggregation)
                unique_clusters = set([m for m in df_profile_clustering_train["membership"] if m != '-1'])
                nb_cluster = len(unique_clusters)
                
                ## ploting only for one plate
                if "Exp06" in plate:
                    ## plot umap with color memebership
                    df_umap_train["membership"] = df_profile_clustering_train["membership"]
                    df_umap_test["membership"] = df_profile_clustering_test["membership"]
                    f1 = px.scatter(df_umap_test, x = "emb1", y = "emb2",
                        color = "membership", hover_data = ["Plate","Well","Treatment"] ,color_discrete_map = color_discrete_map,opacity=0.3, title= f"umap_test_{plate}_leiden_{resolution:.2f}")
                    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

                    ## plot membership aggregated
                    df_grouped = df_profile_clustering_test.groupby(['Treatment', 'membership'], as_index=False).size().rename(columns={"size": "count"})

                    
                    df_grouped = df_grouped.pivot_table(index='Treatment', columns='membership', values='count', fill_value=0)
                    
                    # Normalize by row sums
                    df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

                    df_grouped = df_grouped.stack().reset_index().rename(columns={"level_1": "membership", 0: "percentage"})

                    f1 = px.bar(df_grouped,x = "Treatment", y = "percentage",color = "membership", title = f"agg_Treatment_{plate}_leiden_{resolution:.2f}",color_discrete_map = color_discrete_map)
                    f1.write_html(os.path.join(train_figure_path,f1.layout.__getattribute__("title")["text"] + ".html"))

                    df_grouped.to_csv(os.path.join(base_path_csv, f"agg_Treatment_{plate}_leiden_{resolution:.2f}.csv"))
                
                # Training data aggregation
                agg_mem_train = utils.process_membership_aggregation(
                    df_profile_clustering_train, nb_cluster
                )
                    
                # Test data aggregation
                agg_mem_test = utils.process_membership_aggregation(
                    df_profile_clustering_test, nb_cluster
                )
                
                profiles_plate[f"clustering_leiden_{resolution:.2f}"] = {"train": agg_mem_train, "test": agg_mem_test}



        
        profiles[plate] = profiles_plate
    
    # Save profiles
    print("Saving profiles...")
    pickle.dump(profiles, open(os.path.join(base_path_csv, f"profiles_harmonize_{harmonize}.pkl"), "wb"))
    
    # Calculate similarities and visualize
    print("Calculating profile similarities...")
    similarity_data = utils.calculate_similarities(profiles)
    utils.visualize_similarities(similarity_data, results_figure_path)
    
    # Run classification and visualize
    print("Running classification analysis...")
    classification_data = utils.run_classification(profiles)
    utils.visualize_classification(classification_data, results_figure_path)
    
    # Save final results
    similarity_data.to_csv(os.path.join(base_path_csv, "similarity_metrics.csv"), index=False)
    classification_data.to_csv(os.path.join(base_path_csv, "classification_results.csv"), index=False)
    
    print(f"Pipeline completed. Results saved to {base_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
