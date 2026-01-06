#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import umap


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def feature_sepration(df, feature_type, chans, numeric_cols_metadata):
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
    features_to_use = {chan: [] for chan in chans}
    features_to_use["others"] = []
    for col in features:
        key = "others"
        value_min = np.inf
        for chan in features_to_use.keys():
            val = col.find(chan)
            if val == -1:
                pass
            elif val < value_min:
                value_min = val
                key = chan
        features_to_use[key].append(col)

    features_to_use["all"] = features

    return features_to_use


def feature_seprations_deep(df, chans):
    Fingerprints_to_use = dict()
    for chan in chans:
        Fingerprints_to_use[chan] = df.columns[df.columns.str.contains(f"Feature_{chan}")].tolist()

    Fingerprints_to_use["all"] = []
    for chan in chans:
        Fingerprints_to_use["all"] += Fingerprints_to_use[chan]
    if "BF" in chans:
        Fingerprints_to_use["not-BF"] = list(set(Fingerprints_to_use["all"]).difference(set(Fingerprints_to_use["BF"])))

    return Fingerprints_to_use


def generate_color_map():
    """Generate a color map for visualization."""
    color_discrete_map = {str(i): "rgb" + str(tuple(np.random.choice(256, 3).tolist())) for i in range(-1, 1000)}
    return color_discrete_map


def save_color_map(color_map, params_path):
    """Save the color map to a JSON file."""
    with open(os.path.join(params_path, "color_discrete_map.json"), "w") as f:
        json.dump(color_map, f)


def run_umap(data_train, data_test, seed, metadata, ind_train, ind_val):
    """Run UMAP on training and test data."""
    visualizer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.1, random_state=seed)

    # Train UMAP
    df_umap_train = pd.DataFrame(visualizer.fit_transform(data_train), columns=["emb1", "emb2"])
    df_umap_train = pd.concat([df_umap_train, metadata.loc[ind_train].reset_index(drop=True)], axis=1)

    # Transform test data
    df_umap_test = pd.DataFrame(visualizer.transform(data_test), columns=["emb1", "emb2"])
    df_umap_test = pd.concat([df_umap_test, metadata.loc[ind_val].reset_index(drop=True)], axis=1)

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

