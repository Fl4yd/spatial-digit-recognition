import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
class SimpleTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_depth=5, feature_subset_size='sqrt', random_state=None):
        """
        Simple decision tree classifier for stroke classification.
        
        Args:
            max_depth (int): Maximum depth of the tree
            feature_subset_size (str or int): Number of features to consider at each split.
                If 'sqrt', uses sqrt(n_features)
            random_state (int): Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.feature_subset_size = feature_subset_size
        self.random_state = random_state
        self.tree = None
        self.classes_ = None
        self._rng = np.random.RandomState(self.random_state)
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1 - sum((count / len(y)) ** 2 for count in counts)
        return impurity

    def _get_feature_subset_size(self, n_features):
        """Determine number of features to consider at each split."""
        if self.feature_subset_size == 'sqrt':
            return int(np.sqrt(n_features))
        return min(self.feature_subset_size, n_features)
    
    def _best_split(self, X, y):
        """Find the best split for a node using feature subsets."""
        best_feature, best_threshold, best_gain = None, None, 0
        current_impurity = self._gini_impurity(y)
        
        # Select random subset of features
        n_features = X.shape[1]
        subset_size = self._get_feature_subset_size(n_features)
        feature_indices = self._rng.choice(n_features, size=subset_size, replace=False)
        
        # Only examine the selected features
        for feature in feature_indices:
            # Use quartiles for thresholds instead of all values
            feature_values = X[:, feature]
            thresholds = np.percentile(feature_values, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if min(sum(left_mask), sum(right_mask)) < 2:
                    continue
                
                gain = current_impurity - (
                    (sum(left_mask) / len(y)) * self._gini_impurity(y[left_mask]) +
                    (sum(right_mask) / len(y)) * self._gini_impurity(y[right_mask])
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        if best_feature is None:
            return None, None, 0
        return best_feature, best_threshold, best_gain

    def fit(self, X, y):
        """Fit the tree to training data."""
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        feature, threshold, gain = self._best_split(X, y)
        
        if gain == 0 or feature is None:
            return np.bincount(y).argmax()

        left_mask = X[:, feature] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}

    def predict(self, X):
        """Predict classes for multiple samples."""
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        """Predict class for a single sample."""
        if not isinstance(node, dict):
            return node
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

class SimpleRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=5, random_state=None):
        """
        Simplified Random Forest classifier for stroke classification.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of each tree
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self._rng = np.random.RandomState(self.random_state)
    
    def fit(self, X, y):
        """Train the random forest on the data."""
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Clear any existing trees
        self.trees = []
        
        # Train each tree with bootstrapped samples
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = self._rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create and train tree
            tree = SimpleTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting."""
        # Get predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Compute the mode along the first axis (n_estimators)
        majority_votes, _ = mode(predictions, axis=0)
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
    
    # Pre-allocate arrays with correct size
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
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    # Perform grid search
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate best model on test set
    test_predictions = best_model.predict(X_test)
    
    print("\nBest Model Test Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision (macro): {precision_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    print(f"Recall (macro): {recall_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    print(f"F1 (macro): {f1_score(y_test, test_predictions, average='macro', zero_division=1):.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))
    
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
    
    return best_model

def main():
    # Load data
    df = pd.read_csv('training_data/combined_strokes.csv')
    
    # Prepare data without scaling
    X, y = prepare_data(df, target_length=40)
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.1,
        random_state=42,
        stratify=y
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': np.arange(60, 100, 10),
        'max_depth': np.arange(6, 9)
    }

    #Optimal parameters
    #n_estimators=100, max_depth=9
    
    # Create base classifier
    #base_clf = SimpleRandomForestClassifier(random_state=42)
    
    # Perform grid search
    #best_model = perform_grid_search(
    #    clf=base_clf,
    #    param_grid=param_grid,
    #    X_train=X_train_scaled,
    #    y_train=y_train,
    #    X_test=X_test_scaled,
    #    y_test=y_test
    #)

    best_model = SimpleRandomForestClassifier(
        n_estimators=100,
        max_depth=9,
        random_state=42
    )

    #Test the model
    best_model.fit(X_train_scaled, y_train)
    test_predictions = best_model.predict(X_test_scaled)
    print(confusion_matrix(y_test, test_predictions))
    #Print accuracy
    print(accuracy_score(y_test, test_predictions))

if __name__ == "__main__":
    main()

