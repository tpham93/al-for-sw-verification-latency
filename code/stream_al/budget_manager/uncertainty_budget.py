import numpy as np

class EstimatedBudget:
    def __init__(self, budget, w):
        self.budget = budget
        self.w = w
        self.u_t = 0
    
    def is_budget_left(self):
        return self.u_t/self.w < self.budget
    
    def update_budget(self, sampled):
        for s in sampled:
            self.u_t = self.u_t * (self.w-1.0)/self.w + s

class FixedUncertaintyBudget(EstimatedBudget):
    def __init__(self, budget, w, n_classes):
        super().__init__(budget, w)
        self.n_classes = n_classes
        self.theta = 1/self.n_classes + self.budget * (1-1/self.n_classes)
        
    def query(self, utilities, **kwargs):
        sampled = []
        for u in utilities:
            print((u <= self.theta), self.theta, u)
            sample_instance = super().is_budget_left() and (u <= self.theta)
            super().update_budget([sample_instance])
            sampled.append(sample_instance)
        return np.array(sampled)
    
class VarUncertaintyBudget(EstimatedBudget):
    def __init__(self, budget, w, theta, s):
        super().__init__(budget, w)
        self.theta = theta
        self.s = s
        
    def query(self, utilities, **kwargs):
        sampled = []
        for u in utilities:
            sample_instance = super().is_budget_left()
            if sample_instance:
                sample_instance = u < self.theta
                if sample_instance:
                    self.theta *= 1-self.s
                else:
                    self.theta *= 1+self.s
            super().update_budget([sample_instance])
            sampled.append(sample_instance)
        return np.array(sampled)
    
class SplitBudget(EstimatedBudget):
    def __init__(self, budget, w, rand, v, theta, s):
        super().__init__(budget, w)
        self.rand=rand
        self.v=v
        self.theta = theta
        self.s = s
        
    def query(self, utilities, **kwargs):
        sampled = []
        for u in utilities:
            sample_instance = super().is_budget_left()
            if sample_instance:
                if self.rand.random_sample() < self.v:
                    sample_instance = self.rand.random_sample() < self.budget
                else:
                    sample_instance = u < self.theta
                    if sample_instance:
                        self.theta *= 1-self.s
                    else:
                        self.theta *= 1+self.s
            super().update_budget([sample_instance])
            sampled.append(sample_instance)
        return np.array(sampled)