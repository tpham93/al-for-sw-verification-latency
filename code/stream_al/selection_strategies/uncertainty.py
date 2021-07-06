from .base import BaseSelectionStrategy, StaticBaseSelectionStrategy
from .baseline import RandomSelection
import numpy as np
from copy import deepcopy
from stream_al.util.split_training_data import get_training_data

class Uncertainty(StaticBaseSelectionStrategy):
    def __init__(self, clf_factory_function):
        self.clf_factory_function = clf_factory_function
    
    def utility(self, X, clf, **kwargs):
        tx_n = kwargs['tx_n']
        w_train = kwargs['w_train']
        n_features = kwargs['n_features']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']
        X_n = kwargs['X_n']
        
        modified_training_data = kwargs['modified_training_data']
        
        if modified_training_data:
            add_X = kwargs['add_X']
            add_Y = kwargs['add_Y']
            add_SW = kwargs['add_SW']
            
            new_X_n, Lx_n, Ly_n, Lsw_n = get_training_data(tx_n, w_train, n_features, XT_dict, YT_dict, TY_dict)
            
            if add_X is not None:
                Lx_n = np.concatenate([Lx_n, add_X])
                Ly_n = np.concatenate([Ly_n, add_Y])
                Lsw_n = np.concatenate([Lsw_n, add_SW])
            
            tmp_clf = self.clf_factory_function()
            tmp_clf.fit(Lx_n, Ly_n, X_n, sample_weight=Lsw_n)
        else:
            tmp_clf = clf
        predictions = tmp_clf.predict_proba(X)
        return np.max(predictions, axis=1)

# class FixedUncertainty(StaticBaseSelectionStrategy):
#     def __init__(self, n_classes, budget):
#         self.n_classes = n_classes
#         self.budget = budget
#         self.theta = 1/self.n_classes + self.budget * (1-1/self.n_classes)
        
#     def predict(self, X, clf, **kwargs):
#         pred_proba = np.max(clf.predict_proba(X), axis=1)
#         return pred_proba <= self.theta, self.theta - pred_proba + (1 - self.budget)
    
# class VarUncertainty(BaseSelectionStrategy):
#     def __init__(self, budget, theta_v_u, s_v_u):
#         self.budget = budget
#         self.init_theta_v_u = theta_v_u
#         self.theta_v_u = theta_v_u
#         self.s_v_u = s_v_u
        
#     def predict(self, X, clf, **kwargs):
#         pred_proba = np.max(clf.predict_proba(X), axis=1)
#         return pred_proba <= self.theta_v_u, self.theta_v_u - pred_proba + (1 - self.budget)

#     def reset(self):
#         self.theta_v_u = self.init_theta_v_u
    
#     def partial_fit(self, X, sampled, **kwargs):
#         budget_left = kwargs['budget_left']
#         for s, b in zip(sampled, budget_left):
#             if b:
#                 if s:
#                     self.theta_v_u *= 1-self.s_v_u
#                 else:
#                     self.theta_v_u *= 1+self.s_v_u
            
# class Split(BaseSelectionStrategy):
#     def __init__(self, rand, v, budget, theta_v_u, s_v_u):
#         self.rand = rand
#         self.v = v
#         self.leaners = [RandomSelection(rand, budget), VarUncertainty(budget, theta_v_u, s_v_u)]
#         self.new_learner_indices = []
#         self.budget = budget
        
#     def predict(self, X, clf, **kwargs):
#         num_samples = len(X)
#         learner_indices = self.rand.choice([0, 1], num_samples, p=[self.v, 1-self.v])
#         self.new_learner_indices.extend(learner_indices)
#         output = []
#         output_util = []
#         for (t, x), l in zip(enumerate(X), learner_indices):
#             o, o_u = self.leaners[l].predict([x], clf, **kwargs)
#             output.append(o[0])
#             output_util.append(o_u[0])
#         return output, output_util
            
#     def reset(self):
#         self.new_learner_indices = []
#         for l in self.learners:
#             l.reset()
        
#     def partial_fit(self, X, sampled, **kwargs):
#         budget_left = kwargs['budget_left']
#         for x, s, l, b in zip(X, sampled, self.new_learner_indices, budget_left):
#             partial_fit_kwargs = {
#                 'budget_left':[b]
#             }
#             self.leaners[l].partial_fit([x], [s], **partial_fit_kwargs)