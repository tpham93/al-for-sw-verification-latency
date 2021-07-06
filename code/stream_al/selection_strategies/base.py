from abc import ABC, abstractmethod

class BaseSelectionStrategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        raise NotImplemented
        
    @abstractmethod
    def utility(self, X, **kwargs):
        raise NotImplemented
    
    def fit(self, X, sampled, **kwargs):
        self.reset()
        self.partial_fit(self, X, sampled, **kwargs)
    
    @abstractmethod
    def partial_fit(self, X, sampled, **kwargs):
        raise NotImplemented
    
    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseSelectionStrategy:
            if (any("predict" in B.__dict__ for B in C.__mro__) 
                and any("reset" in B.__dict__ for B in C.__mro__)
                and any("fit" in B.__dict__ for B in C.__mro__)
                and any("partial_fit" in B.__dict__ for B in C.__mro__)):
                return True
            else:
                return NotImplemented
            
class StaticBaseSelectionStrategy(BaseSelectionStrategy):
    def __init__(self):
        super().__init__()
    
    def reset(self):
        pass
    
    @abstractmethod
    def utility(self, x, **kwargs):
        raise NotImplemented
    
    def fit(self, X, sampled, **kwargs):
        return self
    
    def partial_fit(self, x, sampled, **kwargs):
        return self