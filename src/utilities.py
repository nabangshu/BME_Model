import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def metric(x1, y1, x2, y2):
    """
    Computes the MC-RMSE metric (Mean Curve Root Mean Squared Error).
    
    Parameters:
    x1, y1: Coordinates of the first curve.
    x2, y2: Coordinates of the second curve.
    
    Returns:
    float: RMSE-based curve difference.
    """
    temp = np.sort(np.unique(np.union1d(x1, x2)))
    v = np.linspace(temp[0], temp[-1], 10000)

    c1 = np.interp(v, x1, y1)
    c2 = np.interp(v, x2, y2)
    
    area = np.sqrt(np.mean((c1 - c2) ** 2))
    return area


def reconstruct(P, Q, K, curve):
    """
    Reconstructs demand curves.
    
    Parameters:
    P (float): Start point.
    Q (float): Mid point.
    K (float): End point.
    curve (array): Curve data.
    
    Returns:
    numpy array: Reconstructed curve.
    """
    v = np.hstack((0, P, np.linspace(Q, K, len(curve))))
    p = np.hstack((3000, 3000, curve))
    
    return np.vstack((v, p))


def reconstruct_sup(P, Q, K, curve):
    """
    Reconstructs supply curves.
    
    Parameters:
    P (float): Start point.
    Q (float): Mid point.
    K (float): End point.
    curve (array): Curve data.
    
    Returns:
    numpy array: Reconstructed supply curve.
    """
    v = np.hstack((np.linspace(P, Q, len(curve)), K, K + 200))
    p = np.hstack((curve, 3000, 3000))
    
    return np.vstack((v, p))


def model_1(P, public_holidays):
    """
    Trains a Lasso regression model incorporating holiday information.

    Parameters:
    P (numpy array): Input time series data.
    public_holidays (numpy array): Binary vector indicating public holidays.

    Returns:
    Tuple of:
        - Predicted values on validation set
        - Actual test values
        - Predicted values on test set
    """
    P_x = P[6:-1].reshape(-1, 1)
    P_x_7 = P[0:-7].reshape(-1, 1)
    P_y = P[7:].reshape(-1, 1)
    P_x_h = public_holidays[7:].reshape(-1, 1)
    bias = np.ones((len(P_x), 1))

    # Construct feature matrix
    P_covariate = np.hstack((P_x, P_x_7, P_x_h, bias))
    P_target = P_y

    # Split into training, validation, and test sets
    x_train, x_test = P_covariate[:-172, :], P_covariate[-172:, :]
    y_train, y_test = P_target[:-172, :], P_target[-172:, :]
    
    split = int(0.696 * len(x_train))
    x_valid, x_train = x_train[split:], x_train[:split]
    y_valid, y_train = y_train[split:], y_train[:split]

    # Hyperparameter tuning for Lasso regression
    best_alpha, min_error = None, float('inf')
    for alpha in np.linspace(1, 200, 1000):
        lasso = Lasso(alpha=alpha)
        lasso.fit(x_train, y_train)
        y_pred = lasso.predict(x_valid)
        error = np.sqrt(mean_squared_error(y_valid, y_pred))
        if error < min_error:
            min_error = error
            best_alpha = alpha

    # Train Lasso model with best alpha
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(x_train, y_train)

    P_lr = np.hstack((lasso.predict(x_train), lasso.predict(x_valid), lasso.predict(x_test)))
    P_pred_lr = lasso.predict(x_test)
    
    return lasso.predict(x_valid), y_test, P_pred_lr
