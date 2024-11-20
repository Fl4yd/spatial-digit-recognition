import os
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load, parallel_backend
from tqdm import tqdm
class SimpleRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=5, random_state=None, verbose=True):
        """
        Simplified Random Forest classifier for stroke classification.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of each tree
            random_state (int): Random seed for reproducibility
            verbose (bool): Whether to show training progress bar
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []
        self.classes_ = None
        self._rng = np.random.RandomState(self.random_state)
    
    def fit(self, X, y):
        """Train the random forest on the data."""
        # Store unique classes
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        
        # Clear any existing trees
        self.trees = []
        
        # Create iterator based on verbose setting
        iterator = tqdm(range(self.n_estimators), desc="Training trees", unit="tree") if self.verbose else range(self.n_estimators)
        
        # Train each tree with bootstrapped samples
        for _ in iterator:
            # Bootstrap sampling
            indices = self._rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create and train tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting."""
        # Get predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Compute majority votes using numba function
        majority_votes, _ = mode(predictions, axis=0)
        # Map back to class labels
        return majority_votes.ravel().astype(int)

def prepare_data(df, target_length=40):
    """
    Prepare fixed-length samples for classification.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [label, sample, time_step, x, y, z]
        target_length (int): Number of points to interpolate for each sample
        
    Returns:
        X (np.array): Shape (n_samples, target_length * 3) - features
        y (np.array): Shape (n_samples,) - labels
    """
    # Get unique combinations to determine exact array size
    unique_samples = df.groupby(['label', 'sample']).size().reset_index()
    n_samples = len(unique_samples)
    
    # Pre-allocate arrays
    X = np.zeros((n_samples, target_length * 3))
    y = np.zeros(n_samples, dtype=int)
    
    # Process each unique sample with vectorized operations
    for idx, ((label, sample_id), group) in enumerate(df.groupby(['label', 'sample'])):
            
        # Create interpolation points once
        t = group['time_step'].values
        t_new = np.linspace(t.min(), t.max(), target_length)
        
        # Vectorized interpolation for all coordinates
        start_idx = 0
        for coord in ['x', 'y', 'z']:
            X[idx, start_idx:start_idx + target_length] = np.interp(t_new, t, group[coord].values)
            start_idx += target_length
        
        y[idx] = label
    
    return X, y

def perform_grid_search(clf, param_grid, X_train, y_train, X_test, y_test):
    """
    Perform grid search cross-validation and return the best model.

    Args:
        clf: Base classifier
        param_grid (dict): Parameter grid to search
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        best_model: The best performing model
    """
    # Create a new classifier instance with verbose=False for grid search
    base_clf = SimpleRandomForestClassifier(
        random_state=clf.random_state,
        verbose=False  # Disable progress bars during grid search
    )

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=clf.random_state),
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    # Perform grid search
    print("Starting Grid Search...")
    with parallel_backend('loky', n_jobs=-1):
        grid_search.fit(X_train, y_train)
    
    # Ensure all jobs are complete before proceeding
    # Wait for all jobs to finish
    grid_search.best_estimator_.fit(X_train, y_train)
    
    # Print results
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate best model on test set
    test_predictions = best_model.predict(X_test)
    
    # Print top 10 parameter combinations
    print("\nTop 10 parameter combinations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    for idx, row in results_df.head(10).iterrows():
        print(f"\nRank {row['rank_test_score']}:")
        print(f"Parameters: ", end="")
        for param_name in param_grid.keys():
            print(f"{param_name}={row[f'param_{param_name}']}", end=", ")
        print(f"\nMean test score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    print("\nBest Model Validation Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision (macro): {precision_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    print(f"Recall (macro): {recall_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    print(f"F1 (macro): {f1_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    
    
    return best_model

def main():
    # Load data
    df = pd.read_csv('training_data/combined_strokes.csv')

    #Initialize parameters
    target_length = 40
    test_size = 0.1
    random_state = 42
    
    # Prepare data without scaling
    X, y = prepare_data(df, target_length=target_length)
    
    # Split data first into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    #Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.10,
        random_state=random_state,
        stratify=y_train
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': np.arange(60, 110, 10),
        'max_depth': np.arange(5, 10, 1)
    }

    # Create base classifier
    base_clf = SimpleRandomForestClassifier(random_state=42)
    #base_clf.fit(X_train_scaled, y_train)
    # Perform grid search
    best_model = perform_grid_search(
        clf=base_clf,
        param_grid=param_grid,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_val_scaled,
        y_test=y_val
    )
    #best_model = SimpleRandomForestClassifier(random_state=random_state, verbose=True, max_depth=max_depth, n_estimators=n_estimators)
    # Test the model
    #best_model.fit(X_train_scaled, y_train)
    #test_predictions = best_model.predict(X_val_scaled)
    #print(confusion_matrix(y_val, test_predictions))
    # Print accuracy
    #print(accuracy_score(y_val, test_predictions))

    # Test the model on test set
    #best_model.fit(X_train_scaled, y_train)
    test_predictions = best_model.predict(X_test_scaled)
    print(confusion_matrix(y_test, test_predictions))
    # Print accuracy
    print(accuracy_score(y_test, test_predictions))

    # Save model and info
    model_info = {
        'model': best_model,
        'scaler': scaler,
    }
    # Save model and info
    model_path = 'models/stroke_classifier_model.joblib'
    dump(model_info, model_path)

if __name__ == "__main__":
    main()
