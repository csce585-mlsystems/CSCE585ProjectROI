import numpy as np

class Precptron:
    
 def __init__(eta= 0.01, n_iter=50, random_state=1) :
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

def fit(self, x, y) :
  
  regen = np.random.RandomState(self.random_state)
  #self.w = regen.normal(loc=0.0, scale=0.01, sixe=X.shape[1])#decsion boundary remains the same
  self.w = np.zeros(X.shape[1]) # the decsion boundary remains the same

  self.b = np.float(0)
  self.errors_ = []

  for _ in range(self.n_iter) :
    errors = 0
    for xi, target in zip(X,y) :
      update = self.eta * (target - self.predict(xi))
      self.w += update * xi
      self.b += update
      errors += int(update != 0.0)
  self.errors_.append(errors)
  return self 

def net_input(self, X) :
  return np.dot(X, self.w_) + self.b_

def predict(self, X) :
  return np.where(self.net_input (X) >=0.0, 1, 0)
  


