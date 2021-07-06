import numpy as np

LABEL_NOT_SELECTED = -1

def get_training_data(tx_n, w_train, n_features, XT_dict, YT_dict, TY_dict):
    TX = np.array(list(XT_dict.keys()))
    YT = np.array([YT_dict[tx] for tx in TX])
    TY = np.array([TY_dict[tx] for tx in TX])
    
    X_empty = np.zeros(shape=[0, n_features])
    y_empty = np.zeros(shape=[0])
    sampling_weight_empty = np.zeros(shape=[0])
        
    if len(TX):
        map_C_n = TX<tx_n
        map_C_n_selected = np.logical_and(map_C_n, YT!=LABEL_NOT_SELECTED)
        map_C_n_labeled = np.logical_and(map_C_n_selected, TY<tx_n)
        
        Lx_n = np.array([XT_dict[tx_i] for tx_i in TX[map_C_n_labeled]]).reshape([-1, n_features])
            
        Ly_n = np.array([YT_dict[tx_i] for tx_i in TX[map_C_n_labeled]]).reshape([-1])
            
        X_n = np.array([XT_dict[tx_i] for tx_i in TX[map_C_n]]).reshape([-1, n_features])
        
        Lsw_n = np.ones(shape=[len(Lx_n)])
        return X_n, Lx_n, Ly_n, Lsw_n
    else:
        return X_empty, X_empty, y_empty, sampling_weight_empty
        

def get_A_n(tx_n, ty_n, w_train, n_features, XT_dict, YT_dict, TY_dict, delay_future_buffer):
    TX = np.array(list(XT_dict.keys()))
    TX_to_remove = TX[TX < ty_n-w_train+delay_future_buffer]
    
    A_n_XT_dict = XT_dict.copy()
    A_n_YT_dict = YT_dict.copy()
    A_n_TY_dict = TY_dict.copy()
    
    for tx_i in TX_to_remove:
        A_n_XT_dict.pop(tx_i)
        A_n_YT_dict.pop(tx_i)
        A_n_TY_dict.pop(tx_i)
        
    return A_n_XT_dict, A_n_YT_dict, A_n_TY_dict

def get_selected_A_geq_n_tx(tx_n, ty_n, XT_dict, YT_dict, TY_dict):
    TX = np.array(list(XT_dict.keys()))
    YT = np.array([YT_dict[tx] for tx in TX])
    TY = np.array([TY_dict[tx] for tx in TX])
    
    map_A_geq_n = TY >= tx_n
    map_selected = YT!=LABEL_NOT_SELECTED
    map_labeled_until_ty_n = TY < ty_n
    map_A_geq_n = np.logical_and(np.logical_and(map_A_geq_n, map_selected), map_labeled_until_ty_n)
    
    return TX[map_A_geq_n]

def split_training_data(tx_n, ty_n, w_train, XT_dict, YT_dict, TY_dict, n_features, min_timestamp=None):
    TX = np.array(list(XT_dict.keys()))
    TY = np.array([TY_dict[tx] for tx in TX])
    if min_timestamp is not None:
        keys = keys[TX>=min_timestamp]
    if len(TX):
        XT = np.array([XT_dict[tx] for tx in TX]).reshape([-1, n_features])
        YT = np.array([YT_dict[tx] for tx in TX]).reshape([-1])

        map_C_n = TX<tn
        map_A_n = TX>=ty_n-w_train
        A_less_n = np.logical_and(map_C_n, map_A_n)
        A_geq_n = np.logical_and(np.logical_not(map_C_n), map_A_n)
        
        map_not_ST = YT == LABEL_NOT_SELECTED
        keys_not_selected = keys[map_not_ST]
        UT = XT[map_not_ST, :]

        map_ST = np.logical_not(map_not_ST)
        ST_x = XT[map_ST, :]
        ST_y = YT[map_ST]

        map_LT = ST_y != LABEL_SELECTED_BUT_DELAYED
        LT_x = ST_x[map_LT, :]
        LT_y = ST_y[map_LT].astype(float)

        map_DT = np.logical_not(map_LT)
        DT_x = ST_x[map_DT, :]
        
        return LT_x, LT_y, DT_x, UT, XT
    else:
        X_empty = np.zeros(shape=[0, n_features])
        y_empty = np.zeros(shape=[0])
        return X_empty, y_empty, X_empty, X_empty, X_empty