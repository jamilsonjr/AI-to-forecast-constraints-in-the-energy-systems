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
        _error = (_y_test_minus_threshold - _y_pred_minus_threshold)
        _abs_error = abs(_error)
        _sum_abs = abs(_y_test_minus_threshold) + abs(_y_pred_minus_threshold)
        _squared_error = _error ** 2
        true_positives_sse = 0
        self.true_positives_ctr = 0
        false_positives_sse = 0
        self.false_positives_ctr = 0
        false_negatives_sse = 0
        self.false_negatives_ctr = 0
        true_negatives_sse = 0
        self.true_negatives_ctr = 0
        true_positives_sum_abs_error = 0
        true_positives_sum_sum_abs = 0
        false_negatives_sum_abs_error = 0
        false_negatives_sum_sum_abs = 0
        false_positives_sum_abs_error = 0
        false_positives_sum_sum_abs = 0
        true_negatives_sum_abs_error = 0
        true_negatives_sum_sum_abs = 0
        for i in range(_y_test_minus_threshold.shape[0]):
            for j in range(_y_test_minus_threshold.shape[1]):
                if _y_test_minus_threshold.iloc[i, j] > 0:
                    if _y_pred_minus_threshold.iloc[i, j] > 0:
                        true_positives_sse += _squared_error.iloc[i, j]
                        true_positives_sum_abs_error += _abs_error.iloc[i, j]
                        true_positives_sum_sum_abs += _sum_abs.iloc[i, j]
                        self.true_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        false_negatives_sse += _squared_error.iloc[i, j]
                        false_negatives_sum_abs_error += _abs_error.iloc[i, j]
                        false_negatives_sum_sum_abs += _sum_abs.iloc[i, j]
                        self.false_negatives_ctr += 1
                else: #_y_test.iloc[i, j] < 0
                    if _y_pred_minus_threshold.iloc[i, j] > 0:
                        false_positives_sse += _squared_error.iloc[i, j]
                        false_positives_sum_abs_error += _abs_error.iloc[i, j]
                        false_positives_sum_sum_abs += _sum_abs.iloc[i, j]
                        self.false_positives_ctr += 1
                    else: #_y_pred.iloc[i, j] < 0:
                        true_negatives_sse += _squared_error.iloc[i, j]
                        true_negatives_sum_abs_error += _abs_error.iloc[i, j]
                        true_negatives_sum_sum_abs += _sum_abs.iloc[i, j]
                        self.true_negatives_ctr += 1
        # rmse
        compute_rmse = lambda num, den: sqrt(num / den) if den != 0 else 0  
        self.true_positives_rmse = compute_rmse(true_positives_sse, self.true_positives_ctr) 
        self.false_positives_rmse = compute_rmse(false_positives_sse, self.false_positives_ctr)
        self.false_negatives_rmse = compute_rmse(false_negatives_sse, self.false_negatives_ctr)
        self.true_negatives_rmse = compute_rmse(true_negatives_sse, self.true_negatives_ctr)
        # smape
        compute_smape = lambda num, den, ctr:  (1/ctr) * (num / den) if den != 0 else 0
        self.true_positives_smape = compute_smape(true_positives_sum_abs_error, true_positives_sum_sum_abs, self.true_positives_ctr)
        self.false_positives_smape = compute_smape(false_positives_sum_abs_error, false_positives_sum_sum_abs, self.false_positives_ctr)
        self.false_negatives_smape = compute_smape(false_negatives_sum_abs_error, false_negatives_sum_sum_abs, self.false_negatives_ctr)
        self.true_negatives_smape = compute_smape(true_negatives_sum_abs_error, true_negatives_sum_sum_abs, self.true_negatives_ctr)
        # Hybrid Metrics w/ rmse
        self.hybrid_true_positives_rmse = self.true_positives_ctr - (self.true_positives_ctr * self.true_positives_rmse)
        self.hybrid_true_negatives_rmse = self.true_negatives_ctr - (self.true_negatives_ctr * self.true_negatives_rmse)
        self.hybrid_false_positives_rmse = self.false_positives_ctr - (self.false_positives_ctr * self.false_positives_rmse)
        self.hybrid_false_negatives_rmse = self.false_negatives_ctr - (self.false_negatives_ctr * self.false_negatives_rmse)
        # Hybrid Metrics w/ smape
        self.hybrid_true_positives_smape = self.true_positives_ctr - (self.true_positives_ctr * self.true_positives_smape)
        self.hybrid_true_negatives_smape = self.true_negatives_ctr - (self.true_negatives_ctr * self.true_negatives_smape)
        self.hybrid_false_positives_smape = self.false_positives_ctr - (self.false_positives_ctr * self.false_positives_smape)
        self.hybrid_false_negatives_smape = self.false_negatives_ctr - (self.false_negatives_ctr * self.false_negatives_smape)
        # Hybrid Metrics
        compute_recall = lambda tp, fn: tp / (tp + fn) if (tp + fn) != 0 else 0
        compute_precision = lambda tp, fp: tp / (tp + fp) if (tp + fp) != 0 else 0
        compute_f1 = lambda recall, precision: 2 * ((recall * precision) / (recall + precision)) if (recall + precision) != 0 else 0
        compute_accuracy = lambda tp, tn, fp, fn: (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        compute_mcc = lambda tp, tn, fp, fn: ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        # Hybrid Metrics w/ rmse
        self.hybrid_recall_rmse = compute_recall(self.hybrid_true_positives_rmse, self.hybrid_false_negatives_rmse)
        self.hybrid_precision_rmse = compute_precision(self.hybrid_true_positives_rmse, self.hybrid_false_positives_rmse)
        self.hybrid_f1_rmse = compute_f1(self.hybrid_recall_rmse, self.hybrid_precision_rmse)
        self.hybrid_accuracy_rmse = compute_accuracy(self.hybrid_true_positives_rmse, self.hybrid_true_negatives_rmse, self.hybrid_false_positives_rmse, self.hybrid_false_negatives_rmse)
        self.hybrid_mcc_rmse = compute_mcc(self.hybrid_true_positives_rmse, self.hybrid_true_negatives_rmse, self.hybrid_false_positives_rmse, self.hybrid_false_negatives_rmse)
        # Hybrid Metrics w/ smape
        self.hybrid_recall_smape = compute_recall(self.hybrid_true_positives_smape, self.hybrid_false_negatives_smape)
        self.hybrid_precision_smape = compute_precision(self.hybrid_true_positives_smape, self.hybrid_false_positives_smape)
        self.hybrid_f1_smape = compute_f1(self.hybrid_recall_smape, self.hybrid_precision_smape)
        self.hybrid_accuracy_smape = compute_accuracy(self.hybrid_true_positives_smape, self.hybrid_true_negatives_smape, self.hybrid_false_positives_smape, self.hybrid_false_negatives_smape)
        self.hybrid_mcc_smape = compute_mcc(self.hybrid_true_positives_smape, self.hybrid_true_negatives_smape, self.hybrid_false_positives_smape, self.hybrid_false_negatives_smape)
        # Normal metrics
        self.recall = compute_recall(self.true_positives_ctr, self.false_negatives_ctr)
        self.precision = compute_precision(self.true_positives_ctr, self.false_positives_ctr)
        self.f1 = compute_f1(self.recall, self.precision)
        self.accuracy = compute_accuracy(self.true_positives_ctr, self.true_negatives_ctr, self.false_positives_ctr, self.false_negatives_ctr)
        self.mcc = compute_mcc(self.true_positives_ctr, self.true_negatives_ctr, self.false_positives_ctr, self.false_negatives_ctr)
    def print_report(self):
        # Print the above results.
        print('Hybrid Metrics: \n')
        print('True positives RMSPE:', self.true_positives_rmse)
        print('False positives RMSPE:', self.false_positives_rmse)
        print('False negatives RMSPE:', self.false_negatives_rmse)
        print('True negatives RMSPE:', self.true_negatives_rmse)
        print('\n')
        print('True positives hybrid error:', self.hybrid_true_positives_rmse)
        print('False positives hybrid error:', self.hybrid_false_positives_rmse)
        print('False negatives hybrid error:', self.hybrid_false_negatives_rmse)
        print('True negatives hybrid error:', self.hybrid_true_negatives_rmse)
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
    
        