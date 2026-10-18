import math, copy 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

#Load our data set
x_train = np.array([1.0,2.0]) #features
y_train = np.array([300.0,500.0]) #targets

#function to calculate the cost
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost =0.0

    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = cost/(2*m)

    return total_cost

def compute_gradient(x,y,w,b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameters b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw = dj_dw + (f_wb - y[i]) * x[i]
        dj_db = dj_db + f_wb - y[i]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_fucntion):
    """
    performs batch gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with kearning rate alpha

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w_in,b_in (scalar)    : initial model parameters
        alpha (float)         : learning rate
        num_iters (int)       : number of iterations to run gradient descent
        cost_function         : function to call to produce cost
        gradient_function     : function to call to produce gradients

    Returns:
        w (scalar)           : Updated value of parameter after running gradient descent
        b (scalar)           : Updated value of parameter after running gradient descent    
        J_history (list)     : History of cost values
        p_history (list)     : History of parameter values[w, b]
    """
    #An array to store cost J and w's at each iteration primarily for aphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in 

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        #Update parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        #Save cost J at eac iteration 
        if i<100000:
            J_history.append(compute_cost(x,y,w,b))
            p_history.append([w,b])

        #Print cost every interval 10 times or as many iterations if <10
        if i% math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d}:Cost{J_history[-1]:0.2e}",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history
#initialize parameters
w_init = 0
b_init = 0

#some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

#plot cost versus iterations
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True, figsize = (12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration(end)")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')
plt.show()

fig, ax = plt.subplots(1,1, figsize = (12,6))
plt_contour_wgrad(x_train,y_train, p_hist, ax)
plt.show()
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train,y_train, p_hist, ax, w_range =[180, 200, 0.5], b_range = [80, 120, 0.5], contours = [1,5,10,20], resolution = 0.5)
plt.show()


