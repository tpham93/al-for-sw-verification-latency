from .base import BaseSelectionStrategy, StaticBaseSelectionStrategy
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
from copy import deepcopy
from stream_al.util.split_training_data import split_training_data, get_A_n, get_selected_A_geq_n_tx, LABEL_NOT_SELECTED

  
class FOWrapper(BaseSelectionStrategy):
    def __init__(self, random_state, base_selection_strategy, delay_future_buffer):
        self.random_state = random_state
        self.base_selection_strategy = base_selection_strategy
        self.delay_future_buffer = delay_future_buffer
        
    def utility(self, X, clf, **kwargs):
        w_train = kwargs['w_train']
        n_features = kwargs['n_features']
        tx_n = kwargs['tx_n']
        ty_n = kwargs['ty_n']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']
        
        
        new_kwargs = kwargs.copy()
        A_n_XT_dict, A_n_YT_dict, A_n_TY_dict = get_A_n(tx_n, ty_n, w_train, n_features, XT_dict, YT_dict, TY_dict, self.delay_future_buffer)
        
        new_kwargs['XT_dict'] = A_n_XT_dict
        new_kwargs['YT_dict'] = A_n_YT_dict
        new_kwargs['TY_dict'] = A_n_TY_dict
        new_kwargs['modified_training_data'] = True
        
        return self.base_selection_strategy.utility(X, clf, **new_kwargs)
    
    def reset():
        self.base_selection_strategy.reset()
    
    def partial_fit(self, X, sampled, **kwargs):
        self.base_selection_strategy.partial_fit(X, sampled, **kwargs)
    

class BIWrapper(BaseSelectionStrategy):
    def __init__(self, random_state, base_selection_strategy, K, delay_prior, pwc_factory_function):
        self.random_state = random_state
        self.base_selection_strategy = base_selection_strategy
        self.K = K
        self.delay_prior = delay_prior
        self.pwc_factory_function = pwc_factory_function
        
    def get_class_probabilities(self, X, Lx_n, Ly_n, Lsw_n, X_n):
        pwc = self.pwc_factory_function()
        pwc.fit(Lx_n, Ly_n, X_n, sample_weight=Lsw_n)
        frequencies = pwc.predict_freq(X)
        frequencies_w_prior = pwc.predict_freq(X) + self.delay_prior
        probabilities = frequencies_w_prior/np.sum(frequencies_w_prior, axis=1, keepdims=True)
        return probabilities
        
    def utility(self, X, clf, **kwargs):
        tx_n = kwargs['tx_n']
        ty_n = kwargs['ty_n']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']
        
        sum_utilities = np.zeros(len(X))
        
        selected_A_geq_n_tx = get_selected_A_geq_n_tx(tx_n, ty_n, XT_dict, YT_dict, TY_dict)
        if len(selected_A_geq_n_tx):
            A_geq_n_X_selected = np.array([XT_dict[tx_i] for tx_i in selected_A_geq_n_tx]).reshape(len(selected_A_geq_n_tx), -1)
            
            Lx_n = kwargs['Lx_n']
            Ly_n = kwargs['Ly_n']
            Lsw_n = kwargs['Lsw_n']
            X_n = kwargs['X_n']
            predictions = self.get_class_probabilities(A_geq_n_X_selected, Lx_n, Ly_n, Lsw_n, X_n)
            
            for _ in range(self.K):
                A_geq_n_y_prime = np.argmax([self.random_state.multinomial(1, p_d) for p_d in predictions], axis=1)
                new_kwargs = kwargs.copy()
                new_kwargs['add_X'] = A_geq_n_X_selected
                new_kwargs['add_Y'] = A_geq_n_y_prime
                new_kwargs['add_SW'] = np.ones(shape=[len(A_geq_n_y_prime)])
                new_kwargs['modified_training_data'] = True
                utilities = self.base_selection_strategy.utility(X, clf, **new_kwargs)
                sum_utilities += utilities
            return sum_utilities/self.K
        else:
            return self.base_selection_strategy.utility(X, clf, **kwargs)
        return self.base_selection_strategy.utility(X, clf, **kwargs)
    
    def reset():
        self.base_selection_strategy.reset()
    
    def partial_fit(self, X, sampled, **kwargs):
        self.base_selection_strategy.partial_fit(X, sampled, **kwargs)
        

class FIWrapper(BaseSelectionStrategy):
    def __init__(self, random_state, base_selection_strategy, delay_prior, pwc_factory_function):
        self.random_state = random_state
        self.base_selection_strategy = base_selection_strategy
        self.delay_prior = delay_prior
        self.pwc_factory_function = pwc_factory_function
        
    def get_class_probabilities(self, X, Lx_n, Ly_n, Lsw_n, X_n):
        pwc = self.pwc_factory_function()
        pwc.fit(Lx_n, Ly_n, X_n, sample_weight=Lsw_n)
        frequencies = pwc.predict_freq(X)
        frequencies_w_prior = pwc.predict_freq(X) + self.delay_prior
        probabilities = frequencies_w_prior/np.sum(frequencies_w_prior, axis=1, keepdims=True)
        return probabilities
        
    def utility(self, X, clf, **kwargs):
        tx_n = kwargs['tx_n']
        ty_n = kwargs['ty_n']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']
        
        sum_utilities = np.zeros(len(X))
        
        selected_A_geq_n_tx = get_selected_A_geq_n_tx(tx_n, ty_n, XT_dict, YT_dict, TY_dict)
        if len(selected_A_geq_n_tx):
            A_geq_n_X_selected = np.array([XT_dict[tx_i] for tx_i in selected_A_geq_n_tx]).reshape(len(selected_A_geq_n_tx), -1)
            
            Lx_n = kwargs['Lx_n']
            Ly_n = kwargs['Ly_n']
            Lsw_n = kwargs['Lsw_n']
            X_n = kwargs['X_n']
            predictions = self.get_class_probabilities(A_geq_n_X_selected, Lx_n, Ly_n, Lsw_n, X_n)
            
            add_X = []
            add_Y = []
            add_SW = []
            for i in range(len(A_geq_n_X_selected)):
                for c in range(predictions.shape[1]):
                    add_X.append(A_geq_n_X_selected[i])
                    add_Y.append(c)
                    add_SW.append(predictions[i, c])

            new_kwargs = kwargs.copy()
            new_kwargs['add_X'] = np.array(add_X)
            new_kwargs['add_Y'] = np.array(add_Y)
            new_kwargs['add_SW'] = np.array(add_SW)
            new_kwargs['modified_training_data'] = True
            utilities = self.base_selection_strategy.utility(X, clf, **new_kwargs)
            return utilities
        else:
            return self.base_selection_strategy.utility(X, clf, **kwargs)
        return self.base_selection_strategy.utility(X, clf, **kwargs)
    
    def reset():
        self.base_selection_strategy.reset()
    
    def partial_fit(self, X, sampled, **kwargs):
        self.base_selection_strategy.partial_fit(X, sampled, **kwargs)