import numpy as np
from stream_al.selection_strategies.base import BaseSelectionStrategy
import skactiveml.pool._probal as probal
from sortedcontainers import SortedList
from collections import deque
import sklearn

class BIQF:
    def __init__(self, budget, w, w_tol):
        self.budget = budget
        self.w = w
        self.w_tol = w_tol
        self.seen_instances = 0
        self.acquired_instances = 0
        self.historyArr = deque(maxlen=w)
        self.theta_bal = 0
        
    def query(self, utility, **kwargs):
        sampled = []
        
        for i_u, u in enumerate(utility):
            # ipf
            self.seen_instances += 1
            self.historyArr.append(u)
            theta = np.quantile(self.historyArr, (1-self.budget))
            
            # balancing
            range_ranking = np.max(self.historyArr) - np.min(self.historyArr) + 1e-6
            acq_left = self.budget * self.seen_instances - self.acquired_instances
            theta_bal = theta - range_ranking * acq_left / self.w_tol
            
            sample_instance = u >= theta_bal
            
            if sample_instance:
                self.acquired_instances += 1
            sampled.append(sample_instance)
            
            self.theta_bal = theta_bal
        return sampled