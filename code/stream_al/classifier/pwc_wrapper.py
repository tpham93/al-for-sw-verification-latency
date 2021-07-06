import numpy as np
import sklearn

from skactiveml.base import ClassFrequencyEstimator
from skactiveml.classifier import PWC
from sklearn.utils import check_random_state

from sklearn.preprocessing import LabelEncoder

class PWCWrapper(ClassFrequencyEstimator):
    def __init__(self, classes, random_state, N, delta=None, bandwidth='mean'):
        self._random_state = check_random_state(random_state)
        self.classes = classes
        self._le = LabelEncoder().fit(classes)
        self.base_model = PWC(classes=classes, random_state=self._random_state.randint(2**31-1))
        self.base_model.classes_=classes
        self.base_model.cost_matrix_=np.ones([len(classes), len(classes)])
        self.base_model.cost_matrix_[np.eye(len(classes), dtype=int)]=0
        self.base_model.class_prior_ = np.zeros(len(classes))
        if delta is None:
            self.sqr_delta = (np.sqrt(2)*1e-6)**2
        else:
            self.sqr_delta = delta**2
        self.bandwidth = bandwidth
        self.N = N
        
    def fit(self, X, y, X_train, **kwargs):
        # TODO calc N based on budget * (len(sliding_window) - delay) and provide std from the whole sliding window
        self.X_ = X
        
        bandwidth = self.bandwidth
        if self.bandwidth == 'mean':
            bandwidth = 0
            if len(X_train) >= 2:
                sum_sqr_std = np.sum(np.std(X_train, axis=0)**2)
                s = np.sqrt(2*self.N*sum_sqr_std/((self.N-1)*np.log((self.N-1)/self.sqr_delta)))
                if s**2 > 0:
                    bandwidth = 1/(2*s**2)
        self.base_model.metric_dict = {'gamma':bandwidth}
            
        if len(X) > 0:
            self.base_model.fit(X, y, **kwargs)
        return self
        
    def predict_freq(self, X, **kwargs):
        if hasattr(self, 'X_') and len(self.X_):
            return self.base_model.predict_freq(X, **kwargs)
        else:
            return np.zeros([len(X), len(self.classes)])
        
    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        else:
            return getattr(self.base_model, item)
