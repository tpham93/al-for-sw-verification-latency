import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import time

import numpy as np
from  sklearn.datasets import make_blobs

from load_datasets import get_dataset_by_name

from stream_al.util.split_training_data import split_training_data, get_training_data, LABEL_NOT_SELECTED

from stream_al.classifier.pwc_wrapper import PWCWrapper

from stream_al.budget_manager.biqf import BIQF
from stream_al.budget_manager.fixed_budget import FixedBudget
from stream_al.budget_manager.uncertainty_budget import FixedUncertaintyBudget, VarUncertaintyBudget, SplitBudget

from stream_al.selection_strategies.baseline import RandomSelection, PeriodicSample
from stream_al.selection_strategies.uncertainty import Uncertainty
from stream_al.selection_strategies.delay_wrapper import FOWrapper, BIWrapper, FIWrapper
from stream_al.selection_strategies.pal import PAL_BIQF

from sklearn.model_selection import train_test_split

from sacred import Experiment
import pickle

def get_active_learner_by_name(name, **kwargs):
    name_parts = name.split('+')
    first = name_parts[0]
    rest = '+'.join(name_parts[1:])
    
    base_selection_strategy = None
    base_budget_manager = None
    if len(rest):
        base_selection_strategy, base_budget_manager = get_active_learner_by_name(rest, **kwargs)
    
    al_selection_strategies = {}
    al_selection_strategies['rand'] = lambda: (RandomSelection(kwargs['rand']), BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))
    al_selection_strategies['periodic_sample'] = lambda: (PeriodicSample(kwargs['budget']), BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))
  
    al_selection_strategies['fixed_uncer'] = lambda: (Uncertainty(kwargs['clf_factory_function']), FixedUncertaintyBudget(budget=kwargs['budget'], w=kwargs['w'], n_classes=kwargs['n_classes']))
    al_selection_strategies['var_uncer'] = lambda: (Uncertainty(kwargs['clf_factory_function']), VarUncertaintyBudget(budget=kwargs['budget'], w=kwargs['w'], theta=kwargs['theta'], s=kwargs['s']))
    al_selection_strategies['split'] = lambda: (Uncertainty(kwargs['clf_factory_function']), SplitBudget(budget=kwargs['budget'], w=kwargs['w'], rand=kwargs['rand'], v=kwargs['v'], theta=kwargs['theta'], s=kwargs['s']))
    
    al_selection_strategies['pal'] = lambda: (PAL_BIQF(kwargs['pwc_factory_function'], kwargs['prior'], kwargs['m_max']), BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))
    
    al_selection_strategies['FO'] = lambda: (FOWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy, delay_future_buffer=kwargs['delay_future_buffer']), base_budget_manager)
    al_selection_strategies['BI'] = lambda: (BIWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy, K=kwargs['K'], delay_prior=kwargs['delay_prior'], pwc_factory_function=kwargs['pwc_factory_function']), base_budget_manager)
    al_selection_strategies['FI'] = lambda: (FIWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy, delay_prior=kwargs['delay_prior'], pwc_factory_function=kwargs['pwc_factory_function']), base_budget_manager)
    selection_strategy, budget_manager = al_selection_strategies[first]()
    return selection_strategy, budget_manager

ex = Experiment("captured_functions")

@ex.config
def cfg():
    testdate = "2020_08_21_16_00"
    n_reps = 50
    budgetperc = 2
    budget = budgetperc/100.0
    w_train = 500 # changed from 1000
    evaluation_window_size = 1
    
    # clf parameters
    clf_name='PWC'
    # datastream parameters
    dataset_name = 'D0'

    delay_future_buffer = 0
    
    i_rep = 0
    delay = 100
    algo = 'pal'
    prior_exp=0
    K=3
    pkl_output_path = "result_"+testdate+"/d"+str(delay)+"_i"+str(i_rep)+"_a"+algo+"_b"+str(budgetperc)

def get_classifier_factory_functions_by_name(random_state, name, classes, N):
    clf_functions = {}
    clf_functions['PWC'] = lambda: PWCWrapper(classes=classes, random_state=random_state, N=N)
    return clf_functions[name]

@ex.automain
def main(n_reps, budget, w_train, evaluation_window_size,
         clf_name,
         dataset_name,
         delay_future_buffer, prior_exp, K,
         i_rep, delay, algo, pkl_output_path):
    experiment_start_time = time.time()
    dataset_seed = int(dataset_name[1])
    X_stream, y_stream, eval_mask_stream, n_classes, n_features = get_dataset_by_name(dataset_seed, i_rep, dataset_name)
    
    if delay == 'variable':
        delay_random_state = np.random.RandomState(dataset_seed)
        ty_stream = delay_random_state.uniform(50, 300, len(X_stream)) + np.arange(len(X_stream))
        delay = (50+300)/2
    else:
        ty_stream = np.ones(len(X_stream))*delay + np.arange(len(X_stream))
    
    clf_random = i_rep
    pwc_factory_function = get_classifier_factory_functions_by_name(clf_random, 'PWC', np.arange(n_classes), budget*(w_train-delay))
    clf_factory_function = get_classifier_factory_functions_by_name(clf_random, clf_name, np.arange(n_classes), budget*(w_train-delay))
    clf = clf_factory_function()
    XT_dict = {}
    YT_dict = {}
    TY_dict = {}

    label_queue = {}

    rand = np.random.RandomState(i_rep)
    active_learner, budget_manager = get_active_learner_by_name(
        name=algo, 
        rand=rand, 
        budget=budget, 
        n_classes=n_classes, 
        n_features=n_features, 
        theta=1, 
        s=0.01, 
        v=0.1, 
        K=K,
        prior=1e-3,
        m_max=3,
        n_max=5,
        delay_prior=10**prior_exp,
        prior_c=1e-3,
        prior_e=1e-3,
        w=256,
        w_tol=int(200*budget),
        delay_future_buffer=delay_future_buffer,
        pwc_factory_function=pwc_factory_function,
        clf_factory_function=clf_factory_function
    )

    export = {
        'acquisition_timesteps':[],
        'utility':[],
        'prequential_accuracy':[],
#         'sampled':[]
    }

    for tx_n, (x, y, ty_n, e) in enumerate(zip(X_stream, y_stream, ty_stream, eval_mask_stream)):
        if tx_n % 1000 == 0:
            print(tx_n)
        x = x.reshape([1, n_features])
        
        
        X_n, Lx_n, Ly_n, Lsw_n = get_training_data(tx_n, w_train, n_features, XT_dict, YT_dict, TY_dict)
#         LT_x, LT_y, DT_x, UT, XT = split_training_data(XT_dict, YT_dict, n_features)
        clf = clf_factory_function()
        clf.fit(Lx_n, Ly_n, X_n, sample_weight=Lsw_n)
        
        # prequential evaluation
        predicted_class = clf.predict(x)[0]
        prequential_accuracy = int(y == predicted_class)
        
        al_score = active_learner.utility(
            x, 
            clf, 
            X_n=X_n,
            Lx_n=Lx_n, 
            Ly_n=Ly_n,
            Lsw_n=Lsw_n,
            tx_n=tx_n,
            ty_n=ty_n,
            w_train=w_train,
            n_features=n_features,
            XT_dict=XT_dict,
            YT_dict=YT_dict,
            TY_dict=TY_dict,
            modified_training_data=False,
            add_X=None,
            add_Y=None,
            add_SW=None,
        )
        
        if not e:
            sampled = budget_manager.query(al_score)

            active_learner.partial_fit(
                x,
                sampled,
                clf=clf
            )
        else:
            sampled = [False]
        
        if not e:
            XT_dict[tx_n] = x
            TY_dict[tx_n] = ty_n
            if sampled[0]:
                YT_dict[tx_n] = y
                export['acquisition_timesteps'].append(tx_n)
            else:
                YT_dict[tx_n] = LABEL_NOT_SELECTED
            
        t_obsolete = tx_n - w_train
        if t_obsolete >= 0:
            XT_dict.pop(t_obsolete, None)
            YT_dict.pop(t_obsolete, None)
            TY_dict.pop(t_obsolete, None)
            if t_obsolete in label_queue:
                del label_queue[t_obsolete]
        
        experiment_end_time = time.time()
        
        export['utility'].append(al_score)
        export['prequential_accuracy'].append(prequential_accuracy)
        export['time'] = experiment_end_time - experiment_start_time
    
    
    
    with open(pkl_output_path, 'wb') as handle:
        pickle.dump(export, handle, protocol=pickle.HIGHEST_PROTOCOL)
