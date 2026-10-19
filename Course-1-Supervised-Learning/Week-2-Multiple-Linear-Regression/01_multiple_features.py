import numpy as np
import time

#Numpy routines which allocate memory and fill arrays with value
a = np.zeros(4); print(f"np.zeros(4): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,)); print(f"np.zeros((4,)): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")

#Numpy routines which allocate memory and fill arrays with values but do not take shape as input parameter
a = np.arange(4.); print(f"np.arange(4.): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4); print(f"np.random.rand(4): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")

#Numpy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]); print(f"np.array([5,4,3,2]): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a} , a shape = {a.shape}, a data type = {a.dtype}")