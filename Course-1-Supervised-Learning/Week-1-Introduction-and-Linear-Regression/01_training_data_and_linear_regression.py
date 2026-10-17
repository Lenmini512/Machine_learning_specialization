import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#x_train is the input variable (size in 1000 square feet)
#y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0,2.0])
y_train = np.array ([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

#m is the number of training examples
print(f"x_train.shape = {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is : {m}")

#m is the number of training examples
m = len(x_train)
print(f"Number of training examples is : {m}")

i=1
x_i = x_train[i]
y_i = y_train[i]
print(f"x^{i} = {x_i}, y^{i} = {y_i}")

w = 200
b = 100
def compute_model_output(x_train,w,b):
    """
    Computes the output of the model given input x_train and parameters w,b
    Arguments:
    x_train : array of shape (m,) : training data
    w : scalar : weight parameter
    b : scalar : bias parameter
    Returns:
    f_wb : array of shape (m,) : predicted output
    """
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
        
    return f_wb

tmp_f_wb =compute_model_output(x_train,w,b)

#Plot our model predication
plt.plot(x_train, tmp_f_wb, c= 'b', label ='Our Prediction')
plt.scatter(x_train, y_train, marker ='x', c ='r',label ='Actual Values')
plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()

x_i = 1.2
cost_1200sqft = w*x_i + b
print (f"${cost_1200sqft:.0f} thousand dollars")
