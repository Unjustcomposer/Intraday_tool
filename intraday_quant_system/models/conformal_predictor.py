import numpy as np
from typing import List, Set, Tuple

class ConformalPredictor:
    """
    Inductive Conformal Predictor for Binary Classification.
    Provides prediction sets with guaranteed statistical coverage (validity).
    
    Calibration Score:
      Nonconformity score alpha_i = 1 - P(y_i | x_i)
    """
    def __init__(self):
        self.cal_scores_class0 = []
        self.cal_scores_class1 = []
        self.is_calibrated = False

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        """
        Calibrate the predictor using out-of-sample probability outputs.
        
        Args:
            cal_probs: Model probabilities for class 1, shape (N,)
            cal_labels: True binary labels (0 or 1), shape (N,)
        """
        cal_probs = np.clip(cal_probs, 1e-15, 1.0 - 1e-15)
        
        # Calculate nonconformity scores for each class
        # alpha_i = 1 - P(y_i | x_i)
        scores_0 = cal_probs[cal_labels == 0]       # P(1 | x) is the wrong probability for true label 0
        scores_1 = 1.0 - cal_probs[cal_labels == 1] # 1 - P(1 | x) is the wrong probability for true label 1
        
        self.cal_scores_class0 = np.sort(scores_0)
        self.cal_scores_class1 = np.sort(scores_1)
        self.is_calibrated = len(self.cal_scores_class0) > 0 and len(self.cal_scores_class1) > 0
        
    def predict_p_values(self, test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute p-values for y=0 and y=1 for each test probability.
        
        Returns:
            (p0, p1) where:
                p0: p-value for class 0, shape (M,)
                p1: p-value for class 1, shape (M,)
        """
        if not self.is_calibrated:
            # If not calibrated, return uninformative p-values (0.5, 0.5)
            n_samples = len(test_probs)
            return np.ones(n_samples) * 0.5, np.ones(n_samples) * 0.5
            
        test_probs = np.clip(test_probs, 1e-15, 1.0 - 1e-15)
        
        n0 = len(self.cal_scores_class0)
        n1 = len(self.cal_scores_class1)
        
        p0 = []
        p1 = []
        
        for p in test_probs:
            # Nonconformity score if true label was 0
            alpha_new_0 = p
            # Nonconformity score if true label was 1
            alpha_new_1 = 1.0 - p
            
            # Count calibration scores greater than or equal to alpha_new
            count_0 = np.sum(self.cal_scores_class0 >= alpha_new_0)
            count_1 = np.sum(self.cal_scores_class1 >= alpha_new_1)
            
            # Calculate p-value: (count + 1) / (N + 1)
            p_val_0 = (count_0 + 1) / (n0 + 1)
            p_val_1 = (count_1 + 1) / (n1 + 1)
            
            p0.append(p_val_0)
            p1.append(p_val_1)
            
        return np.array(p0), np.array(p1)

    def predict_set(self, test_probs: np.ndarray, significance_level: float = 0.05) -> List[Set[int]]:
        """
        Return conformal prediction sets at significance level epsilon.
        The prediction set contains labels with p-value > significance_level.
        
        Args:
            test_probs: Model probabilities for class 1, shape (M,)
            significance_level: Epsilon (e.g. 0.05 for 95% confidence)
        """
        p0, p1 = self.predict_p_values(test_probs)
        pred_sets = []
        
        for idx in range(len(test_probs)):
            labels = set()
            if p0[idx] > significance_level:
                labels.add(0)
            if p1[idx] > significance_level:
                labels.add(1)
            pred_sets.append(labels)
            
        return pred_sets
        
    def get_signal_confidence(self, test_prob: float, significance_level: float = 0.05) -> Tuple[str, float]:
        """
        Synthesize conformal prediction into a signal and confidence metric.
        
        Returns:
            (conformal_signal, confidence) where:
                conformal_signal: 'buy' (set is {1}), 'sell' (set is {0}), or 'no_trade' (set is {0,1} or {})
                confidence: credibility score (max p-value) or confidence score (1 - second max p-value)
        """
        p0, p1 = self.predict_p_values(np.array([test_prob]))
        p0_val, p1_val = p0[0], p1[0]
        
        # Conformal Set
        in_0 = p0_val > significance_level
        in_1 = p1_val > significance_level
        
        if in_1 and not in_0:
            signal = 'buy'
            # Confidence: 1 - p-value of the wrong class
            confidence = 1.0 - p0_val
        elif in_0 and not in_1:
            signal = 'sell'
            confidence = 1.0 - p1_val
        else:
            signal = 'no_trade'
            confidence = 0.0
            
        return signal, confidence
