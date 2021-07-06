import numpy as np
from stream_al.selection_strategies.base import BaseSelectionStrategy, StaticBaseSelectionStrategy
import skactiveml.pool._probal as probal
from sortedcontainers import SortedList
from collections import deque
import sklearn
from copy import deepcopy
from stream_al.util.split_training_data import split_training_data, get_training_data, LABEL_NOT_SELECTED


class PAL_BIQF(StaticBaseSelectionStrategy):
    def __init__(self, pwc_factory_function, prior, m_max, n_max = None):
        self.pwc_factory_function = pwc_factory_function
        self.prior = prior
        self.m_max = m_max
        self.n_max = n_max
        
    def utility(self, X, clf, **kwargs):
        tx_n = kwargs['tx_n']
        w_train = kwargs['w_train']
        n_features = kwargs['n_features']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']
        modified_training_data = kwargs['modified_training_data']
        
        if modified_training_data:
            add_X = kwargs['add_X']
            add_Y = kwargs['add_Y']
            add_SW = kwargs['add_SW']
            
            X_n, Lx_n, Ly_n, Lsw_n = get_training_data(tx_n, w_train, n_features, XT_dict, YT_dict, TY_dict)
            
            if add_X is not None:
                Lx_n = np.concatenate([Lx_n, add_X])
                Ly_n = np.concatenate([Ly_n, add_Y])
                Lsw_n = np.concatenate([Lsw_n, add_SW])
            
            tmp_clf = self.pwc_factory_function()
            tmp_clf.fit(Lx_n, Ly_n, X_n, sample_weight=Lsw_n)
        else:
            if hasattr(clf, 'predict_freq'):
                tmp_clf = clf
            else:
                tmp_clf = self.pwc_factory_function()
                tmp_clf.fit(kwargs['Lx_n'], kwargs['Ly_n'], kwargs['X_n'], sample_weight=kwargs['Lsw_n'])
        
        k_vec = tmp_clf.predict_freq(X)
        n = np.sum(k_vec)
        if self.n_max is not None and n > self.n_max:
            k_vec = k_vec/n * self.n_max
        utilities = probal._cost_reduction(k_vec, prior=self.prior, m_max=self.m_max)
        return utilities
    
# class Delayed_PAL_BIQF_OLD(BaseSelectionStrategy):
#     def __init__(self, prior, m_max, n_max = None, delay_prior = 0):
#         self.prior = prior
#         self.m_max = m_max
#         self.n_max = n_max
#         self.delay_prior = delay_prior
        
#     def predict(self, X, clf, **kwargs):
#         X_train_unlabeled = kwargs['X_train_unlabeled']
#         X_train_all = kwargs['X_train_all']
#         k_vec = clf.predict_freq(X)
# #         n = np.sum(k_vec, axis=1)
#         k_vec_prior = k_vec + self.delay_prior
#         n_prior = np.sum(k_vec_prior, axis=1)
#         if n_prior > 0:
#             p = k_vec_prior/n_prior
#         else:
#             p = k_vec * 0 + (1/k_vec.shape[1])
        
#         if X_train_unlabeled.shape[0] > 0:
#             clf_d = deepcopy(clf).fit(X_train_unlabeled, np.zeros(len(X_train_unlabeled)), X_train_all)
#             n_d = np.sum(clf_d.predict_freq(X), axis=1)
#             k_vec += n_d * p
            
#         n = np.sum(k_vec)
#         if self.n_max is not None and n > self.n_max:
#             k_vec = k_vec/n * self.n_max
            
#         utilities = probal.cost_reduction(k_vec, prior=self.prior, m_max=self.m_max)
#         return utilities, utilities

#     def reset(self):
#         pass
    
#     def partial_fit(self, X, sampled, clf, **kwargs):
#         pass

# class Delayed_PAL_BIQF(BaseSelectionStrategy):
#     def __init__(self, prior_c, prior_e, m_max):
#         self.prior_c = prior_c
#         self.prior_e = prior_e
#         self.m_max = m_max
#         self.density = 0
        
#     def predict(self, X, clf, **kwargs):
#         # get kwargs arguments
#         X_train_filtered = kwargs['X_train_filtered']
#         X_train_unlabeled = kwargs['X_train_unlabeled']
#         X_unlabeled = kwargs['X_unlabeled']
#         X_train_all = kwargs['X_train_all']
        
#         # calculate frequencies based on all labeled instances
#         k_L_mat = clf.predict_freq(X)
        
#         # calculate the n for all unlabeled instances (used for density)
#         tmp_clf = deepcopy(clf).fit(X_unlabeled, np.zeros(len(X_unlabeled)), X_train_all)
#         n_unlabeled = np.sum(tmp_clf.predict_freq(X), axis=1)
        
#         # calculate k vector for labeled instances and selected but still unlabeled instances
#         if len(X_train_unlabeled):
#             # add a prior to the predicted class probabilities
#             freq_X_train_unlabeled = clf.predict_freq(X_train_unlabeled) + self.prior_c
# #             proba_X_train_unlabeled = freq_X_train_unlabeled/np.sum(freq_X_train_unlabeled)
#             proba_X_train_unlabeled = freq_X_train_unlabeled/np.sum(freq_X_train_unlabeled, axis=1, keepdims=True)
    
#             # construct new classifier with fuzzy labeling by including each instance with each class but setting the weigths to the estimated class probability
#             X_train_unlabeled_with_weights = []
#             weights = []
#             y_train_unlabeled_with_weights = []
#             for x, y_proba_vec in zip(X_train_unlabeled, proba_X_train_unlabeled):
#                 for c, y_proba in enumerate(y_proba_vec):
#                     X_train_unlabeled_with_weights.append(x)
#                     y_train_unlabeled_with_weights.append(c)
#                     weights.append(y_proba)
#             clf_d = tmp_clf.fit(X_train_unlabeled_with_weights, y_train_unlabeled_with_weights, X_train_all, sample_weight=weights)
#             k_Ld_mat = k_L_mat + clf_d.predict_freq(X)
#         else:
#             k_Ld_mat = k_L_mat
#         n_Ld = np.sum(k_Ld_mat, axis=1)
        
#         if len(X) > 1:
#             raise NotImplementedError()
#         else:
#             # density is currently ignored, thus, set to 1
#             # +1 is for X itself
#             density_vec = 1 + 0 *(n_unlabeled + n_Ld + 1)/(len(X_unlabeled) + len(X_train_filtered) + len(X_train_unlabeled) + 1)
# #             density_vec = [len(X_unlabeled) + len(X_train_filtered) + len(X_train_unlabeled) + 1]
# #             density_vec = [len(X_unlabeled)]
        
#         self.density = density_vec[0]
#         utilities = []
#         for k_L, k_Ld, density in zip(k_L_mat, k_Ld_mat, density_vec):
#             utilities.append(delayed_pal(k_L, k_Ld, density, self.m_max, self.prior_c, self.prior_e))
        
#         return utilities, utilities

#     def reset(self):
#         pass
    
#     def partial_fit(self, X, sampled, clf, **kwargs):
#         pass    
    
# class Delayed_PAL_BIQF_BOOTSTRAP(BaseSelectionStrategy):
#     def __init__(self, random_state, prior_c, prior_e, m_max, num_bootstraps):
#         self.prior_c = prior_c
#         self.prior_e = prior_e
#         self.m_max = m_max
#         self.density = 0
#         self.random_state = random_state
#         self.num_bootstraps = num_bootstraps
        
#     def predict(self, X, clf, **kwargs):
#         # get kwargs arguments
#         X_train_filtered = kwargs['X_train_filtered']
#         X_delayed = kwargs['X_train_unlabeled']
#         X_unlabeled = kwargs['X_unlabeled']
#         X_train_all = kwargs['X_train_all']
        
#         # calculate frequencies based on all labeled instances
#         k_L_mat = clf.predict_freq(X)
        
#         # calculate the n for all unlabeled instances (used for density)
#         tmp_clf = deepcopy(clf).fit(X_unlabeled, np.zeros(len(X_unlabeled)), X_train_all)
#         n_unlabeled = np.sum(tmp_clf.predict_freq(X), axis=1)
        
#         utilities = np.zeros([len(X), self.m_max, self.num_bootstraps])
        
        
#         for i_bootstrap in range(self.num_bootstraps):
#             # calculate k vector for labeled instances and selected but still unlabeled instances
#             if len(X_delayed):
#                 # add a prior to the predicted class probabilities
#                 freq_X_delayed = clf.predict_freq(X_delayed) + self.prior_c
#     #             proba_X_delayed = freq_X_delayed/np.sum(freq_X_delayed)
#                 proba_X_delayed = freq_X_delayed/np.sum(freq_X_delayed, axis=1, keepdims=True)

#                 #TODO
#                 bootstrapped_y_delayed = np.argmax([self.random_state.multinomial(1, p_d) for p_d in proba_X_delayed], axis=1)

#                 clf_d = tmp_clf.fit(X_delayed, bootstrapped_y_delayed, X_train_all)
#                 k_Ld_mat = k_L_mat + clf_d.predict_freq(X)
#             else:
#                 k_Ld_mat = k_L_mat
#             n_Ld = np.sum(k_Ld_mat, axis=1)

#             if len(X) > 1:
#                 raise NotImplementedError()
#             else:
#                 # density is currently ignored, thus, set to 1
#                 # +1 is for X itself
#                 density_vec = 1 + 0 *(n_unlabeled + n_Ld + 1)/(len(X_unlabeled) + len(X_train_filtered) + len(X_delayed) + 1)
#     #             density_vec = [len(X_unlabeled) + len(X_train_filtered) + len(X_delayed) + 1]
#     #             density_vec = [len(X_unlabeled)]

#             self.density = density_vec[0]
#             for i_x, (k_L, k_Ld, density) in enumerate(zip(k_L_mat, k_Ld_mat, density_vec)):
#                 for m in range(1, self.m_max+1):
#                     utilities[i_x, m-1, i_bootstrap] = delayed_pal2(k_L, k_Ld, density, m, self.prior_c, self.prior_e)
        
#         utilities = np.max(np.mean(utilities, axis=2), axis=1)
#         return utilities, utilities

#     def reset(self):
#         pass
    
#     def partial_fit(self, X, sampled, clf, **kwargs):
#         pass    

class Delayed_PAL_BIQF_FUTURE(BaseSelectionStrategy):
    def __init__(self, random_state, prior_c, prior_e, m_max, num_bootstraps, delay, n_features, delay_future_buffer = 0):
        self.prior_c = prior_c
        self.prior_e = prior_e
        self.m_max = m_max
        self.density = 0
        self.random_state = random_state
        self.delay = delay
        self.num_bootstraps = num_bootstraps
        self.n_features = n_features
        self.delay_future_buffer = delay_future_buffer
        
    def predict(self, X, clf, **kwargs):
        # get kwargs arguments
        t = kwargs['t']
        max_size_train = kwargs['max_size_train']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        UT = kwargs['UT']
        DT_x = kwargs['DT_x']
        XT = kwargs['XT']
        
        LF_x, LF_y, DF_x, UF, XF = split_training_data(XT_dict, YT_dict, self.n_features, t-max_size_train+self.delay+self.delay_future_buffer)
        
        kX_LT = clf.predict_freq(X)
        
        # calculate frequencies based on all labeled instances
        tmp_clf = deepcopy(clf)
        tmp_clf.fit(LF_x, LF_y, XT)
        kX_LF = tmp_clf.predict_freq(X)
        
        utilities = np.zeros([len(X), self.m_max, self.num_bootstraps])
        
        for i_bootstrap in range(self.num_bootstraps):
            # calculate k vector for labeled instances and selected but still unlabeled instances
            if len(DF_x):
                # add a prior to the predicted class probabilities
                kDT_x_LT = clf.predict_freq(DT_x) + self.prior_c
                proba_DT_x_LT = kDT_x_LT/np.sum(kDT_x_LT, axis=1, keepdims=True)
                
                YT_Bootstrap = np.argmax([self.random_state.multinomial(1, p_d) for p_d in proba_DT_x_LT], axis=1)
                tmp_clf.fit(DT_x, YT_Bootstrap, XT)
                kX_LT_Y_Bootstrap = kX_LT + tmp_clf.predict_freq(X)
                
                tmp_clf.fit(DT_x[-len(DF_x):], YT_Bootstrap[-len(DF_x):], XT)
                kX_LF_Y_Bootstrap = kX_LF + tmp_clf.predict_freq(X)
            else:
                kX_LT_Y_Bootstrap = kX_LT
                kX_LF_Y_Bootstrap = kX_LF
                
            # calculate the n for all unlabeled instances (used for density)
            tmp_clf.fit(UT, np.zeros(len(UT)), XT)
            n_unlabeled = np.sum(tmp_clf.predict_freq(X), axis=1)
            n_Ld = np.sum(kX_LT_Y_Bootstrap, axis=1)
            if len(X) > 1:
                raise NotImplementedError()
            else:
                # density is currently ignored, thus, set to 1
                # +1 is for X itself
                density_vec = 1 + 0 *(n_unlabeled + n_Ld + 1)/(len(XT) + 1)
                # density_vec = [len(X_unlabeled) + len(X_train_filtered) + len(X_delayed) + 1]
                # density_vec = [len(X_unlabeled)]
            
            
            for i_x, (k_LT, k_LT_Y_Bootstrap, k_LF, k_LF_Y_Bootstrap, density) in enumerate(zip(kX_LT, kX_LT_Y_Bootstrap, kX_LF, kX_LF_Y_Bootstrap, density_vec)):
                for m in range(1, self.m_max+1):
                    utilities[i_x, m-1, i_bootstrap] = delayed_pal_future(k_LT, k_LT_Y_Bootstrap, k_LF, k_LF_Y_Bootstrap, density, m, self.prior_c, self.prior_e)
        
        utilities = np.max(np.mean(utilities, axis=2), axis=1)
        return utilities, utilities

    def reset(self):
        pass
    
    def partial_fit(self, X, sampled, clf, **kwargs):
        pass    

class Delayed_PAL_BIQF_FUZZY(BaseSelectionStrategy):
    def __init__(self, random_state, prior_c, prior_e, m_max, delay, n_features, delay_future_buffer = 0):
        self.prior_c = prior_c
        self.prior_e = prior_e
        self.m_max = m_max
        self.density = 0
        self.random_state = random_state
        self.delay = delay
        self.n_features = n_features
        self.delay_future_buffer = delay_future_buffer
        
    def predict(self, X, clf, **kwargs):
        # get kwargs arguments
        t = kwargs['t']
        max_size_train = kwargs['max_size_train']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        UT = kwargs['UT']
        DT_x = kwargs['DT_x']
        XT = kwargs['XT']
        
        LF_x, LF_y, DF_x, UF, XF = split_training_data(XT_dict, YT_dict, self.n_features, t-max_size_train+self.delay+self.delay_future_buffer)
        
        kX_LT = clf.predict_freq(X)
        
        # calculate frequencies based on all labeled instances
        tmp_clf = deepcopy(clf)
        tmp_clf.fit(LF_x, LF_y, XT)
        kX_LF = tmp_clf.predict_freq(X)

        utilities = np.zeros([len(X), self.m_max])
        
        # calculate k vector for labeled instances and selected but still unlabeled instances
        if len(DF_x):
            # estimate class probabilities for delayed instances
            kDT_x_LT = clf.predict_freq(DT_x) + self.prior_c
            proba_DT_x_LT = kDT_x_LT/np.sum(kDT_x_LT, axis=1, keepdims=True)
            
            # copy each instance n_class-times to use for fuzzy labeling
            n_classes = proba_DT_x_LT.shape[1]
            DT_per_class = np.tile(DT_x, [1, n_classes]).reshape(-1, DT_x.shape[1])
            # assign each instance each possible class
            YT_Fuzzy = np.tile(np.arange(n_classes), proba_DT_x_LT.shape[0]).reshape(-1)
            # set the weight for each instance label combination to the respective probability
            sample_weight = proba_DT_x_LT.reshape(-1)
            
            tmp_clf.fit(DT_per_class, YT_Fuzzy, XT, sample_weight=sample_weight)
            kX_LT_Y_Fuzzy = kX_LT + tmp_clf.predict_freq(X)
            
            tmp_clf.fit(DT_per_class[-(len(DF_x)*n_classes):], YT_Fuzzy[-(len(DF_x)*n_classes):], XT, sample_weight=sample_weight[-(len(DF_x)*n_classes):])
            kX_LF_Y_Fuzzy = kX_LF + tmp_clf.predict_freq(X)
        else:
            kX_LT_Y_Fuzzy = kX_LT
            kX_LF_Y_Fuzzy = kX_LF

        # calculate the n for all unlabeled instances (used for density)
        tmp_clf.fit(UT, np.zeros(len(UT)), XT)
        n_unlabeled = np.sum(tmp_clf.predict_freq(X), axis=1)
        n_Ld = np.sum(kX_LT_Y_Fuzzy, axis=1)
        if len(X) > 1:
            raise NotImplementedError()
        else:
            # density is currently ignored, thus, set to 1
            # +1 is for X itself
            density_vec = 1 + 0 *(n_unlabeled + n_Ld + 1)/(len(XT) + 1)
            # density_vec = [len(X_unlabeled) + len(X_train_filtered) + len(X_delayed) + 1]
            # density_vec = [len(X_unlabeled)]


        for i_x, (k_LT, k_LT_Y_Fuzzy, k_LF, k_LF_Y_Fuzzy, density) in enumerate(zip(kX_LT, kX_LT_Y_Fuzzy, kX_LF, kX_LF_Y_Fuzzy, density_vec)):
            for m in range(1, self.m_max+1):
                utilities[i_x, m-1] = delayed_pal_future(k_LT, k_LT_Y_Fuzzy, k_LF, k_LF_Y_Fuzzy, density, m, self.prior_c, self.prior_e)
        
        utilities = np.max(utilities, axis=1)
        return utilities, utilities

    def reset(self):
        pass
    
    def partial_fit(self, X, sampled, clf, **kwargs):
        pass    
    
# def delayed_pal(k_L, k_Ld, density, m_max, prior_c, prior_e):
#     k_L = list(k_L)
#     k_Ld = list(k_Ld)
#     gain = []
#     n_classes = len(k_L)
#     argmax = lambda k: max(list(range(n_classes)), key=lambda i: k[i])
#     pred_old = argmax(k_Ld)
#     sum_p_yc =  sum(k_L) + prior_c * n_classes

#     for m in range(1, m_max+1):
#         gain_m = 0
#         sum_p_ye = m + sum(k_Ld) + prior_e * n_classes
#         for yc in range(n_classes):
#             p_yc = (k_L[yc] + prior_c)**m
#             k_Ld_x = list(k_Ld)
#             k_Ld_x[yc] += m
#             delta_acc = 0
#             for ye in range(n_classes):
#                 p_ye = (k_Ld_x[ye] + prior_e)
#                 loss_old = pred_old == ye
#                 loss_new = argmax(k_Ld_x) == ye
#                 delta_acc += p_ye * (int(loss_new) - int(loss_old)) 
#             gain_m += p_yc * delta_acc
#         gain.append(gain_m / m / (sum_p_yc**m) / sum_p_ye)
#     return density * max(gain)
            
# def delayed_pal2(k_L, k_Ld, density, m, prior_c, prior_e):
#     k_L = list(k_L)
#     k_Ld = list(k_Ld)
#     gain = []
#     n_classes = len(k_L)
#     argmax = lambda k: max(list(range(n_classes)), key=lambda i: k[i])
#     pred_old = argmax(k_Ld)
#     sum_p_yc =  sum(k_L) + prior_c * n_classes

#     gain_m = 0
#     sum_p_ye = m + sum(k_Ld) + prior_e * n_classes
#     for yc in range(n_classes):
#         p_yc = (k_L[yc] + prior_c)**m
#         k_Ld_x = list(k_Ld)
#         k_Ld_x[yc] += m
#         delta_acc = 0
#         for ye in range(n_classes):
#             p_ye = (k_Ld_x[ye] + prior_e)
#             loss_old = pred_old == ye
#             loss_new = argmax(k_Ld_x) == ye
#             delta_acc += p_ye * (int(loss_new) - int(loss_old)) 
#         gain_m += p_yc * delta_acc
#     gain.append(gain_m / m / (sum_p_yc**m) / sum_p_ye)
#     return density * max(gain)

# def delayed_pal_future(k_L, k_Ld, k_L_future, k_Ld_future, density, m, prior_c, prior_e):
def delayed_pal_future(k_LT, k_LT_Y_B, k_LF, k_LF_Y_B, density, m, prior_c, prior_e):
    k_LT = list(k_LT)
    k_LT_Y_B = list(k_LT_Y_B)
    k_LF = list(k_LF)
    k_LF_Y_B = list(k_LF_Y_B)
    n_classes = len(k_LT)
    argmax = lambda k: max(list(range(n_classes)), key=lambda i: k[i])
    pred_old_future = argmax(k_LF_Y_B)
    sum_p_yc =  sum([(k+prior_c)**m for k in k_LT])

    gain_m = 0
    sum_p_ye = m + sum(k_LT_Y_B) + prior_e * n_classes
    for yc in range(n_classes):
        p_yc = (k_LT[yc] + prior_c)**m
        k_LT_Y_B_m = list(k_LT_Y_B)
        k_LT_Y_B_m[yc] += m
        k_LF_Y_B_m = list(k_LF_Y_B)
        k_LF_Y_B_m[yc] += m
        delta_acc = 0
        for ye in range(n_classes):
            p_ye = (k_LT_Y_B_m[ye] + prior_e)
            loss_old = pred_old_future == ye
            loss_new = argmax(k_LF_Y_B_m) == ye
            delta_acc += p_ye * (int(loss_new) - int(loss_old)) 
        gain_m += p_yc * delta_acc
    
    gain = gain_m / m / sum_p_yc / sum_p_ye
    return density * gain
    
# class Delayed_PAL_BIQF(BaseSelectionStrategy):
#     def __init__(self, prior, m_max):
#         self.prior = prior
#         self.m_max = m_max
        
#     def predict(self, X, clf, **kwargs):
#         X_train_unlabeled = kwargs['X_train_unlabeled']
#         k_vec = clf.predict_freq(X)
        
#         if X_train_unlabeled.shape[0] > 0:
#             proba_x_unlabeled = clf.predict_proba(X_train_unlabeled)
#             y_unlabeled = np.arange(X_train_unlabeled.shape[0])
#             clf_resp = deepcopy(clf)
#             clf_resp.classes = y_unlabeled
#             clf_resp.fit(X_train_unlabeled, y_unlabeled)
#             resp = clf_resp.predict_freq(X)
#             print(X_train_unlabeled.shape)
#             print(resp.shape)
#             print(proba_x_unlabeled.shape)
#             k_vec += resp @ proba_x_unlabeled
#         utilities = probal.cost_reduction(k_vec, prior=self.prior, m_max=self.m_max)
#         return utilities, utilities

#     def reset(self):
#         pass
    
#     def partial_fit(self, X, sampled, clf, **kwargs):
#         pass