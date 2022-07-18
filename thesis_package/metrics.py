import matplotlib.pyplot as plt
from numpy import sqrt
import numpy as np
class Metrics:
    def get_prediction_scores(self, y_pred, y_test, threshold):
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
        self.true_positives_ctr = 0
        false_positives_sse = 0
        self.false_positives_ctr = 0
        false_negatives_sse = 0
        self.false_negatives_ctr = 0
        true_negatives_sse = 0
        self.true_negatives_ctr = 0
        for i in range(_y_test.shape[0]):
            for j in range(_y_test.shape[1]):
                if _y_test.iloc[i, j] > 0:
                    if _y_pred.iloc[i, j] > 0:
                        true_positives_sse += _squared_error.iloc[i, j]
                        self.true_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        false_negatives_sse += _squared_error.iloc[i, j]
                        self.false_negatives_ctr += 1
                else: #_y_test.iloc[i, j] < 0
                    if _y_pred.iloc[i, j] > 0:
                        false_positives_sse += _squared_error.iloc[i, j]
                        self.false_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        true_negatives_sse += _squared_error.iloc[i, j]
                        self.true_negatives_ctr += 1
        self.true_positives_rmse = sqrt(true_positives_sse/self.true_positives_ctr)
        self.false_positives_rmse = sqrt(false_positives_sse/self.false_positives_ctr)
        self.false_negatives_rmse = sqrt(false_negatives_sse/self.false_negatives_ctr)
        self.true_negatives_rmse = sqrt(true_negatives_sse/self.true_negatives_ctr)
    def get_report(self):
        # Print the above results.
        print('True positives RMSE:', self.true_positives_rmse)
        print('False positives RMSE:', self.false_positives_rmse)
        print('False negatives RMSE:', self.false_negatives_rmse)
        print('True negatives RMSE:', self.true_negatives_rmse)
        # Compute recall from the above results.
        # Compute recall from the above results.
        if (self.true_positives_ctr + self.false_negatives_ctr) != 0:
            self.recall = (self.true_positives_ctr) / (self.true_positives_ctr + self.false_negatives_ctr)
        else:
            self.recall = 0
        print('Recall:', self.recall, '\n', 'TP: {}, FN: {}'.format(self.true_positives_ctr, self.false_negatives_ctr))
        # Compute accuracy from the above results.
        if self.true_positives_ctr + self.true_negatives_ctr + self.false_positives_ctr + self.false_negatives_ctr != 0:
            self.accuracy = (self.true_positives_ctr + self.true_negatives_ctr) / (self.true_positives_ctr + self.true_negatives_ctr + self.false_positives_ctr + self.false_negatives_ctr)
        else:
            self.accuracy = 0
        print('Accuracy:', self.accuracy, '\n', 'TP: {}, FP: {}, TN: {}, FN: {}'.format(self.true_positives_ctr, self.false_positives_ctr, self.true_negatives_ctr, self.false_negatives_ctr))
        # Compute precision of the above results.
        if self.true_positives_ctr + self.false_positives_ctr != 0:
            self.precision = (self.true_positives_ctr) / (self.true_positives_ctr + self.false_positives_ctr)
        else: 
            self.precision = 0
        print('Precision:', self.precision, '\n', 'TP: {}, FP: {}'.format(self.true_positives_ctr, self.false_positives_ctr))
        # Compute F1 score from the above results.
        if self.precision + self.recall != 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0
        print('F1 score:', self.f1_score, '\n', 'Precision: {}, Recall: {}'.format(self.precision, self.recall))
    def plot_series(self, series1, series2, threshold=None, title=None):
        plt.figure(figsize=(25,8))
        plt.plot(series1, label='Ground truth')
        plt.plot(series2, label='Predictions')
        if threshold is not None:
            plt.plot(threshold, 'r', label='Threshold')
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.show()
    
