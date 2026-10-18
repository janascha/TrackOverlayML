"""
Dataset classes for Track Overlay ML model.

This module provides dataset wrappers that:
1. Extract features from preprocessed dataframes
2. Extract TargetLabel values
3. Manage feature lists consistently
4. Provide dataset containers used by training/evaluation
"""

import pandas as pd
import numpy as np
from data.features import get_feature_names


class xRIMdataset:
    """
    Dataset wrapper for track overlay features and labels.

    Extracts the selected ML features and TargetLabel values from a
    preprocessed dataframe.

    Feature scaling is intentionally not performed here. The scaler is
    fitted on the training sample only in loadTrainValTestData() and then
    applied unchanged to validation, test, and evaluation samples.
    
    Attributes:
        df: Input dataframe with track features and TargetLabel
        use_densities: Whether to include density/event features
        features: Raw, unscaled feature array (N_tracks x N_features)
        featureList: Names of features
        labels: Binary labels (1=Diff, 0=NoDiff)
    """
    
    def __init__(self, df, logfile=None, use_densities=False):
        self.use_densities = use_densities
        self.df = df
        self.features, self.featureList = self.getFeatures(logfile=logfile)
        self.labels = self.getLabels()
        self.shape = self.features.shape

    def getFeatures(self, logfile=None):
        """
        Extract raw track features from dataframe.
        
        Features include:
        - Basic kinematics: Px, Py, Pz, E, Pt
        - Densities (if use_densities=True): R02, R05, sumR02, sumR05, sumpT02, sumpT05
        - Event-level: w_numPU, TruthMultiplicity, eventPt
        
        Returns:
            features: Raw, unscaled feature array
            feat_list: List of feature names
        """
        
        feat_list = get_feature_names(self.use_densities)
        features = self.df[feat_list].to_numpy(dtype=float)
        
        # Log feature ranges before scaling
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        print("Feature Ranges (before scaling):", file=logfile, flush=True)
        for i, name in enumerate(feat_list):
            print(f"  {name}: min={feature_min[i]:.6f}, max={feature_max[i]:.6f}", 
                  file=logfile, flush=True)
        
        return features, feat_list

    def getLabels(self):
        """
        Extract binary labels from dataframe.
        
        Returns:
            labels: 1D array
                1 = Diff: MC Overlay and Track Overlay disagree on matching
                0 = NoDiff: MC Overlay and Track Overlay agree on matching
        """
        labels = self.df[["TargetLabel"]].values.squeeze()
        return labels


class bundlexRIM:
    """
    Container for training and validation datasets with class weights.
    
    Bundles train/test splits together with their associated class weights
    for balanced training. Primarily used for legacy compatibility.
    
    Attributes:
        training: xRIMdataset for training
        validation: xRIMdataset for validation/testing
        train_classW: Class weights for training set
        val_classW: Class weights for validation set
        trainStats: Number of training samples
        valStats: Number of validation samples
    """
    
    def __init__(self, training=None, validation=None, W_train=None, W_test=None, logfile=None):
        self.training = training if training is not None else []
        self.validation = validation if validation is not None else []
        self.trainStats = training.shape[0] if training is not None else 0
        self.valStats = validation.shape[0] if validation is not None else 0
        self.train_classW = W_train if W_train is not None else []
        self.val_classW = W_test if W_test is not None else []

