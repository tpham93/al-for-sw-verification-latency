class FixedBudget:
    def __init__(self, budget):
        self.budget = budget
        self.seen_instances = 0
        self.acquired_instances = 0
    
    def query(self, utility):
        sampled = []
        # utility for zliobaite returns only True/False
        for u in utility:
            self.seen_instances += 1
            if self.budget * self.seen_instances - self.acquired_instances >= 1:
                self.acquired_instances += u
                budget_left.append(True)
                sampled.append(u)
            else:
                budget_left.append(False)
                sampled.append(False)
                
        return sampled