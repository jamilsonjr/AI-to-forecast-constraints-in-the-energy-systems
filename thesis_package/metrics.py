import matplotlib.pyplot as plt
from numpy import sqrt
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
        _y_test_minus_threshold = y_test.reset_index(drop=True) - threshold
        _y_pred_minus_threshold = y_pred - threshold
        _squared_error = (_y_test_minus_threshold - _y_pred_minus_threshold) ** 2
        true_positives_sse = 0
        self.true_positives_ctr = 0
        false_positives_sse = 0
        self.false_positives_ctr = 0
        false_negatives_sse = 0
        self.false_negatives_ctr = 0
        true_negatives_sse = 0
        self.true_negatives_ctr = 0
        for i in range(_y_test_minus_threshold.shape[0]):
            for j in range(_y_test_minus_threshold.shape[1]):
                if _y_test_minus_threshold.iloc[i, j] > 0:
                    if _y_pred_minus_threshold.iloc[i, j] > 0:
                        true_positives_sse += _squared_error.iloc[i, j]
                        self.true_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        false_negatives_sse += _squared_error.iloc[i, j]
                        self.false_negatives_ctr += 1
                else: #_y_test.iloc[i, j] < 0
                    if _y_pred_minus_threshold.iloc[i, j] > 0:
                        false_positives_sse += _squared_error.iloc[i, j]
                        self.false_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        true_negatives_sse += _squared_error.iloc[i, j]
                        self.true_negatives_ctr += 1
        compute_rmse = lambda num, den: sqrt(num / den) if den != 0 else 0  
        self.true_positives_rmspe = compute_rmse(true_positives_sse, self.true_positives_ctr) 
        self.false_positives_rmspe = compute_rmse(false_positives_sse, self.false_positives_ctr)
        self.false_negatives_rmspe = compute_rmse(false_negatives_sse, self.false_negatives_ctr)
        self.true_negatives_rmspe = compute_rmse(true_negatives_sse, self.true_negatives_ctr)
        # Hybrid Metrics 
        self.true_positives_hybrid_error = self.true_positives_ctr - (self.true_positives_ctr * self.true_positives_rmspe)
        self.false_positives_hybrid_error = self.false_positives_ctr - (self.false_positives_ctr * self.false_positives_rmspe)
        self.false_negatives_hybrid_error = self.false_negatives_ctr - (self.false_negatives_ctr * self.false_negatives_rmspe)
        self.true_negatives_hybrid_error = self.true_negatives_ctr - (self.true_negatives_ctr * self.true_negatives_rmspe)
        if (self.true_positives_hybrid_error + self.false_negatives_hybrid_error) != 0:
            self.hybrid_recall = self.true_positives_hybrid_error / (self.true_positives_hybrid_error + self.false_negatives_hybrid_error)
        else: 
            self.hybrid_recall = 0
        if (self.true_positives_hybrid_error + self.false_positives_hybrid_error) != 0:
            self.hybrid_precision = self.true_positives_hybrid_error / (self.true_positives_hybrid_error + self.false_positives_hybrid_error)
        else:
            self.hybrid_precision = 0
        if (self.hybrid_precision + self.hybrid_recall) != 0:
            self.hybrid_f1 = 2 * (self.hybrid_precision * self.hybrid_recall) / (self.hybrid_precision + self.hybrid_recall)
        else:
            self.hybrid_f1 = 0
        if self.true_positives_hybrid_error + self.true_negatives_hybrid_error + self.false_positives_hybrid_error + self.false_negatives_hybrid_error != 0:
            self.hybrid_accuracy = (self.true_positives_hybrid_error + self.true_negatives_hybrid_error) / (self.true_positives_hybrid_error + self.false_negatives_hybrid_error + self.true_negatives_hybrid_error + self.false_positives_hybrid_error)
        else:
            self.hybrid_accuracy = 0
        # Normal metrics
        # Precision, recall, accuracy, f1 score.
        if (self.true_positives_ctr + self.false_negatives_ctr) != 0:
            self.recall = (self.true_positives_ctr) / (self.true_positives_ctr + self.false_negatives_ctr)
        else:
            self.recall = 0
        # print('Recall:', self.recall, '\n', 'TP: {}, FN: {}'.format(self.true_positives_ctr, self.false_negatives_ctr))
        # Compute accuracy from the above results.
        if self.true_positives_ctr + self.true_negatives_ctr + self.false_positives_ctr + self.false_negatives_ctr != 0:
            self.accuracy = (self.true_positives_ctr + self.true_negatives_ctr) / (self.true_positives_ctr + self.true_negatives_ctr + self.false_positives_ctr + self.false_negatives_ctr)
        else:
            self.accuracy = 0
        # print('Accuracy:', self.accuracy, '\n', 'TP: {}, FP: {}, TN: {}, FN: {}'.format(self.true_positives_ctr, self.false_positives_ctr, self.true_negatives_ctr, self.false_negatives_ctr))
        # Compute precision of the above results.
        if self.true_positives_ctr + self.false_positives_ctr != 0:
            self.precision = (self.true_positives_ctr) / (self.true_positives_ctr + self.false_positives_ctr)
        else: 
            self.precision = 0
        # print('Precision:', self.precision, '\n', 'TP: {}, FP: {}'.format(self.true_positives_ctr, self.false_positives_ctr))
        # Compute F1 score from the above results.
        if self.precision + self.recall != 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0
    def print_report(self):
        # Print the above results.
        print('Hybrid Metrics: \n')
        print('True positives RMSPE:', self.true_positives_rmspe)
        print('False positives RMSPE:', self.false_positives_rmspe)
        print('False negatives RMSPE:', self.false_negatives_rmspe)
        print('True negatives RMSPE:', self.true_negatives_rmspe)
        print('\n')
        print('True positives hybrid error:', self.true_positives_hybrid_error)
        print('False positives hybrid error:', self.false_positives_hybrid_error)
        print('False negatives hybrid error:', self.false_negatives_hybrid_error)
        print('True negatives hybrid error:', self.true_negatives_hybrid_error)
        print('\n')
        print('Hybrid recall:', self.hybrid_recall)
        print('Hybrid precision:', self.hybrid_precision)
        print('Hybrid F1:', self.hybrid_f1)
        print('Hybrid accuracy:', self.hybrid_accuracy)
        print('\n')
        print('Normal metrics: \n')
        print('True positives:', self.true_positives_ctr)
        print('False positives:', self.false_positives_ctr)
        print('False negatives:', self.false_negatives_ctr)
        print('True negatives:', self.true_negatives_ctr)
        print('\n')
        print('Recall:', self.recall)
        print('Accuracy:', self.accuracy)
        print('Precision:', self.precision)
        print('F1 score:', self.f1_score)

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
    
        