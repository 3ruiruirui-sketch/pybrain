#!/usr/bin/env python3
"""
DenseWolf-K: Binary Grey Wolf Optimizer (BGWO) for Feature Selection
=====================================================================
Optimizes the 183-feature radiomics pipeline by selecting the most 
informative feature subset.

Mathematical Core: Binary Grey Wolf Optimizer (BGWO)
Fitness: 0.99 * ErrorRate + 0.01 * (NumSelectedFeatures / TotalFeatures)
Classifier: K-Nearest Neighbors (KNN=5) with 5-Fold Cross-Validation.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_radiomics.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "gwo_optimal_features.json"

N_WOLVES = 15
MAX_ITER = 30
KNN_NEIGHBORS = 5
CV_FOLDS = 5
N_FEATURES_TOTAL = 183

def banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)

# --- DATA HANDLING (load_data) -----------------------------------------------
def load_data():
    """
    Load radiomics data or generate robust synthetic dataset (400 samples, 183 features).
    Exactly mimics the pipeline output size.
    """
    if DATA_PATH.exists():
        print(f"  📂 Loading radiomics dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=['target']).values
        y = df['target'].values
        feature_names = df.drop(columns=['target']).columns.tolist()
    else:
        print("  ⚠️ Dataset not found. Generating robust 183-feature synthetic radiomics data...")
        X, y = make_classification(
            n_samples=400, 
            n_features=N_FEATURES_TOTAL, 
            n_informative=15, 
            n_redundant=20, 
            random_state=42
        )
        feature_names = [f"feature_{i:03d}" for i in range(N_FEATURES_TOTAL)]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names

# --- BGWO CORE ---------------------------------------------------------------
class BinaryGreyWolfOptimizer:
    def __init__(self, X, y, n_wolves=15, max_iter=30):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        
        # Initialize wolf positions [N_WOLVES, n_features]
        # Binary arrays (1 = select, 0 = drop)
        self.positions = np.random.randint(2, size=(n_wolves, self.n_features)).astype(np.float32)
        
        # Best wolves (Alpha, Beta, Delta) - Objective is MINIMIZATION
        self.alpha_pos = np.zeros(self.n_features)
        self.alpha_score = float('inf')  # Initialize with infinity
        
        self.beta_pos = np.zeros(self.n_features)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(self.n_features)
        self.delta_score = float('inf')

    def calculate_fitness(self, mask):
        """
        Evaluate a feature mask using KNN-5CV.
        Fitness = 0.99 * ErrorRate + 0.01 * (NumSelectedFeatures / TotalFeatures)
        Minimizing this value optimizes both accuracy and compactness.
        """
        selected = np.where(mask > 0.5)[0]
        
        # Handle empty selection
        if len(selected) == 0:
            return 1.0 # Maximum error penalty
            
        X_selected = self.X[:, selected]
        knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
        
        # 5-fold cross validation for accuracy
        try:
            scores = cross_val_score(knn, X_selected, self.y, cv=CV_FOLDS)
            accuracy = scores.mean()
        except Exception:
            # Fallback if SVM/KNN fails on extreme subspaces
            return 1.0

        error_rate = 1.0 - accuracy
        feature_ratio = len(selected) / self.n_features
        
        # Mathematical Fitness Score (Minimization objective)
        fitness = 0.99 * error_rate + 0.01 * feature_ratio
        return fitness

    def transfer_function(self, x):
        """
        Sigmoid Transfer Function: T(x) = 1 / (1 + exp(-10 * (x - 0.5)))
        Used to map continuous consensus positions back into binary space.
        """
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

    def hunt(self):
        banner(f"BGWO: HUNTING THE GOLDEN FEATURE SUBSET (183)")
        
        for t in range(self.max_iter):
            # 1. Update Fitness and find Alpha, Beta, Delta (Predatory leaders)
            for i in range(self.n_wolves):
                fitness = self.calculate_fitness(self.positions[i])
                
                # Check for Alpha (Best)
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                
                # Check for Beta (Second Best)
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                
                # Check for Delta (Third Best)
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            # 2. Update wolves' positions based on leaders
            a = 2 - t * (2 / self.max_iter) # a decreases linearly from 2 to 0
            
            for i in range(self.n_wolves):
                for j in range(self.n_features):
                    # Tracking Alpha
                    r1 = random.random(); r2 = random.random()
                    A1 = 2 * a * r1 - a; C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Tracking Beta
                    r1 = random.random(); r2 = random.random()
                    A2 = 2 * a * r1 - a; C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Tracking Delta
                    r1 = random.random(); r2 = random.random()
                    A3 = 2 * a * r1 - a; C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Consensus position update [Mirjalili et al., 2014]
                    X_consensus = (X1 + X2 + X3) / 3.0
                    
                    # Sigmoid Transfer to binary space
                    if random.random() < self.transfer_function(X_consensus):
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0
            
            # Record Accuracy (inverse of error rate within fitness) for logging
            alpha_accuracy = 1.0 - ((self.alpha_score - 0.01 * (self.alpha_pos.sum() / self.n_features)) / 0.99)
            print(f"  Iteration {t+1:02d}/{self.max_iter} — pack consensus: {100*alpha_accuracy:.1f}% Acc | {int(self.alpha_pos.sum())}/{self.n_features} Feats")

        return self.alpha_pos, self.alpha_score

# --- MAIN --------------------------------------------------------------------
def main():
    banner("DENSE-WOLF-K: BINARY GREY WOLF OPTIMIZER")
    
    # 1. Load data (mimicking pipeline output)
    X, y, feature_names = load_data()
    print(f"  Samples: {X.shape[0]} | Target Features: {X.shape[1]}")
    
    # 2. Run BGWO HUNT
    bgwo = BinaryGreyWolfOptimizer(X, y, n_wolves=N_WOLVES, max_iter=MAX_ITER)
    best_mask, best_fitness = bgwo.hunt()
    
    # 3. Process Results
    selected_indices = np.where(best_mask > 0.5)[0].tolist()
    selected_names = [feature_names[i] for i in selected_indices]
    
    # Calculate final accuracy
    final_acc = 1.0 - ((best_fitness - 0.01 * (len(selected_names) / N_FEATURES_TOTAL)) / 0.99)
    
    print("\n" + "═" * 70)
    print("  HUNT COMPLETE")
    print(f"  Best Feature Set: {len(selected_names)} items")
    print(f"  Final Accuracy: {final_acc*100:.2f}%")
    print(f"  Final Minimization Score: {best_fitness:.6f}")
    print("═" * 70)
    
    # 4. Save results (Golden Feature List)
    output_data = {
        "algorithm": "Binary Grey Wolf Optimizer (BGWO)",
        "objective": "Minimization of 0.99*Error + 0.01*FeatureRatio",
        "timestamp": str(pd.Timestamp.now()),
        "final_fitness": best_fitness,
        "final_accuracy": final_acc,
        "n_selected": len(selected_names),
        "total_candidate_features": N_FEATURES_TOTAL,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "boolean_mask": (best_mask > 0.5).tolist()
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"  💾 Golden Feature List saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
