#Indexing means referring to an element of an array by its position within the array.
#Slicing means getting a subset of elements from an array based on their indices.

import numpy as np
import time
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access and element
print(f"a[2].shape = {a[2].shape}, a[2] ={a[2]}, accessing an element return a scalar")

#access the last elemnt, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print(f"The error message you will see is :")
    print(e)

#Slicing

#vector slicing operations
a = np.arange(10)
print(f"a = {a}")

#access five consecutive elemnts (start:stop:step)
c = a[2:7:1] ; print("a[2:7:1] = ", c ) #It always avoid printing the stop number we are giving 

#access three elements seperated by two
c = a[2:7:2] ; print("a[2:7:2]", c)

#access all elements index 3 and above
c = a[3:] ; print("a[3:]", c)

#access all the elements below 3
c = a[:3] ; print("a[:3]", c)

#access all elements
c = a[:] ; print("a[:]", c)

#Single vector opoerations

a = np.array([1,2,3,4])
print("a              : ", a)

#negative elements of a
b = -a
print("b= -a          : ", b)

#sum all elements of a, returns a scalar
b = np.sum(a)
print("np.sum(a)      : ", b)

b = np.mean(a)
print("np.mean(a)     : ", b)

b = a**2
print("a**2           : ", b)

#vector vector element-wise operations
a = np.array([1,2,3,4])
b = np.array([-1,-2,3,4])
print(f"Binary operators work element wise: {a+b}")

#try a mismatched vector operation
c = np.array([1,2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is :")
    print(e)

#Scalar vector operations
a = np.array([1, 2, 3, 4])

#multiply a by a scalar
b = 5*a
print(f"b = 5 * a : {b}")

def my_dot(a,b):
    """
    Compute the dot product of two vectors

    Args:
      a (ndarray (n,)): input vector
      b (ndarray (n,)) : input vector with same dimension as a

    Returns:
      x (scalar) :
    """
    x=0
    for i in range(a.shape[0]):
     x = x +a[i]*b[i]
    return x
#test 1-d
a = np.array([1, 2, 3, 4])
a = np.array([-1, 4, 3, 2])
print(f"my_dot(a,b) = {my_dot(a,b)}")

c = np.dot(a,b)
print(f"Numpy i D np.dot(a,b) = {c}, np.dot(a,b).shape = {c.shape}")
c = np.dot(b,a)
print(f"Numpy i D np.dot(b,a) = {c}, np.dot(b,a).shape = {c.shape}")

np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(f"np.dot(a,b) = {c:4f}")
print(f"Vectorized version duration: {1000*(toc-tic):4f} ms")


tic = time.time()
c = my_dot(a,b)
toc = time.time()

print(f"my_dot(a,b) = {c:4f}")
print(f"Loop version duration: {1000*(toc-tic):4f} ms")

del(a);del(b)

# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] = {X[1]} ,X[1] has shape {X[1].shape}")
print(f"w = {w}, w has shape {w.shape}")
print(f"c = {c}, c has shape {c.shape}")

#MATRICES
#m is the number of rows (training data examples)
#n is the number of columns (feature number)

a = np.zeros((1,5))
print(f"a shape ={a.shape}, a = {a}")
a = np.zeros((2,1))
print(f"a shape ={a.shape}, a = {a}")
a = np.random.random_sample((1,1))
print(f"a shape ={a.shape}, a = {a}")

#matrix creation
#Numpy routines which allocates memory and fill with user specified values
a = np.array([[5],[4],[3]]); print(f"a shape ={a.shape}, a = {a}")
a = np.array([[5],
              [4],
              [3]]) ; 
print(f"a shape ={a.shape}, a = {a}")

#indexing 
#vector indexing operations on matrices
a = np.arange(6).reshape(-1,2) #(-1)means it automatically calculates the required rows #only one dimension can use -1
print(f"a shape ={a.shape}, \na = {a}")

#access an element
print(f"\na[2,0].shape: {a[2,0].shape}, a[2,0] = {a[2,0]}, type(a[2,0]) = {type(a[2,0])}")

#access a row
print(f"\na[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}")

#slicing 
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")





