"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # random number generator for Fixed seeds (to make sure we can genarate same thing during each run)
    rng = np.random.default_rng(seed)
    
    #make permutations
    indices = rng.permutation(X.shape[0])
    
    # Same indices apply to both X and y
    return X[indices], y[indices]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n = len(y )
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train = list(range(0,n_train))
    val = list(range(n_train,n_train+n_val))
    test = list(range(n_train+n_val,n))

    return X[train] , y[train] ,X[val] , y[val],X[test] , y[test]

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    
    mean = np.mean(X, axis=0) 
    std = np.std(X, axis=0)  

    std[std == 0 ] = 1 #repace 0 by 1 , for prevent from zero div erro  
    return mean , std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    X_n = (X - mean)/std
    return X_n

# Step 5 - add_bias_column
def add_bias_column(X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    X = np.hstack([ones,X])

    return X

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X = standardize_features(X, mean, std)
    return  add_bias_column(X)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    
    return np.dot(X,weights)

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    a = y_true - y_pred
    a = a**2
    return np.sum(a)/(y_true.shape[0])

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    N = len(X)
    y_erro = y_true - y_pred 
    

    return (-2/N)*(X.T)@y_erro

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    b  = (X.T)@X
    a = X.T@y

    return np.linalg.solve(b,a)

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

