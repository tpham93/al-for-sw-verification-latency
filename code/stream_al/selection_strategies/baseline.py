from .base import BaseSelectionStrategy, StaticBaseSelectionStrategy

import numpy as np

class RandomSelection(StaticBaseSelectionStrategy):
    def __init__(self, rand):
        super().__init__()
        self.rand = rand
    
    def utility(self, X, clf, **kwargs):
        output = self.rand.random_sample(len(X))
        return output

class PeriodicSample(BaseSelectionStrategy):
    def __init__(self, budget):
        super().__init__()
        self.samples_per_instance = budget
        self.used_budget = 0
        self.seen_instances = 0
    
    def utility(self, X, clf, **kwargs):
        output = np.zeros(len(X))
        bought = 0
        for t, x in enumerate(X):
            num_seen = self.seen_instances + (t+1)
            num_bought = self.used_budget + bought
            output[t] = num_seen * self.samples_per_instance - num_bought
            if output[t] >= 1:
                bought += 1
        return output * 0
    
    def reset(self):
        self.used_budget = 0
        self.seen_instances = 0
    
    def partial_fit(self, X, sampled, **kwargs):
        for s in sampled:
            if s:
                self.used_budget += 1
            self.seen_instances += 1
        return self