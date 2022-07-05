import matplotlib.pyplot as plt
from numpy import sqrt

def get_prediction_scores(y_pred, y_test, threshold):
    """ Function that returns a hybrid metrics
        Args:
            y_pred, y_test: pd.Dataframe().
            activation_threshold: float.
        Returns:
            pd.DataFrame: Dataframe with the power flow profiles.
        """
    # Computation.
    _y_test = y_test.reset_index(drop=True) - threshold
    _y_pred = y_pred - threshold
    _squared_error = (_y_test - _y_pred) ** 2
    true_positives_sse = 0
    true_positives_ctr = 0
    false_positives_sse = 0
    false_positives_ctr = 0
    false_negatives_sse = 0
    false_negatives_ctr = 0
    true_negatives_sse = 0
    true_negatives_ctr = 0
    for i in range(_y_test.shape[0]):
        for j in range(_y_test.shape[1]):
            if _y_test.iloc[i, j] > 0:
                if _y_pred.iloc[i, j] > 0:
                    true_positives_sse += _squared_error.iloc[i, j]
                    true_positives_ctr += 1
                else: #_y_pred.iloc[i, j] < 0:
                    false_negatives_sse += _squared_error.iloc[i, j]
                    false_negatives_ctr += 1
            else: #_y_test.iloc[i, j] < 0
                if _y_pred.iloc[i, j] > 0:
                    false_positives_sse += _squared_error.iloc[i, j]
                    false_positives_ctr += 1
                else: #_y_pred.iloc[i, j] < 0:
                    true_negatives_sse += _squared_error.iloc[i, j]
                    true_negatives_ctr += 1
    true_positives_rmse = sqrt(true_positives_sse/true_positives_ctr)
    false_positives_rmse = sqrt(false_positives_sse/false_positives_ctr)
    false_negatives_rmse = sqrt(false_negatives_sse/false_negatives_ctr)
    true_negatives_rmse = sqrt(true_negatives_sse/true_negatives_ctr)
    return true_positives_rmse, false_positives_rmse, false_negatives_rmse, true_negatives_rmse, true_positives_ctr, false_positives_ctr, false_negatives_ctr, true_negatives_ctr
def plot_series(series1, series2, title):
    plt.figure(figsize=(25,8))
    plt.plot(series1, label='Ground truth')
    plt.plot(series2, label='Predictions')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.show()
