"""
Dataset classes for Track Overlay ML model.

This module provides dataset wrappers that:
1. Load features from preprocessed dataframes
2. Extract labels (MatchProb-based)
3. Apply MinMax scaling to features
4. Manage train/validation splits with class weights
"""

import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from data.features import get_feature_names


class xRIMdataset:
    """
    Dataset wrapper for track overlay features and labels.
    
    Loads features from preprocessed dataframe, extracts labels based on
    MatchProb, and applies MinMax scaling for neural network training.
    
    Attributes:
        df: Input dataframe with track features and TargetLabel
        use_densities: Whether to include R02/R05 density features
        features: Scaled feature array (N_tracks x N_features)
        featureList: Names of features
        labels: Binary labels (1=good match, 0=poor match)
        fscaler: Fitted MinMaxScaler for features
    """
    
    def __init__(self, df, logfile=None, use_densities=False):
        self.use_densities = use_densities
        self.df = df
        self.features, self.featureList = self.getFeatures(logfile=logfile)
        self.labels = self.getLabels()
        self.shape = self.features.shape

    def getFeatures(self, logfile=None):
        """
        Extract and scale track features from dataframe.
        
        Features include:
        - Basic kinematics: Px, Py, Pz, E, Pt
        - Densities (if use_densities=True): R02, R05, sumR02, sumR05, sumpT02, sumpT05
        - Event-level: w_numPU, TruthMultiplicity, eventPt
        
        Returns:
            features: Scaled feature array (MinMax normalized to [0, 1])
            feat_list: List of feature names
        """
        
        feat_list = get_feature_names(self.use_densities)
        features = self.df[feat_list].to_numpy(dtype=float)
        
        #feat = 14 if self.use_densities else 5
        #feat_list = ["px", "py", "pz", "E", "$P_T$"]
        #features = np.zeros((self.df.shape[0], feat))

        # Basic kinematics
        #features[:, 0] = self.df["Px"]
        #features[:, 1] = self.df["Py"]
        #features[:, 2] = self.df["Pz"]
        #features[:, 3] = self.df["E"]
        #features[:, 4] = self.df["Pt"]
        
        # Add density features if requested
        #if self.use_densities:
            #features[:, 5] = self.df["R02"]
            #features[:, 6] = self.df["R05"]
            #features[:, 7] = self.df["sumR02"]
            #features[:, 8] = self.df["sumR05"]
            #features[:, 9] = self.df["sumpT02"]
            #features[:,10] = self.df["sumpT05"]
            #features[:,11] = self.df["w_numPU"]
            #features[:,12] = self.df["TruthMultiplicity"]
            #features[:,13] = self.df["eventPt"]
            #feat_list += ["R02", "R05", "sumR02", "sumR05", "sumpT02", "sumpT05", 
            #              "w_numPU", "TruthMultiplicity", "eventPt"]
        
        # Log feature ranges before scaling
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        print("Feature Ranges (before scaling):", file=logfile, flush=True)
        for i, name in enumerate(feat_list):
            print(f"  {name}: min={feature_min[i]:.6f}, max={feature_max[i]:.6f}", 
                  file=logfile, flush=True)
        
        # Apply MinMax scaling to [0, 1]
        #Fscaler = MinMaxScaler()
        #self.fscaler = Fscaler
        #features = Fscaler.fit_transform(features)
        
        return features, feat_list

    def getLabels(self):
        """
        Extract binary labels from dataframe.
        
        Returns:
            labels: 1D array (1=good match MatchProb>0.5, 0=poor match)
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

