import numpy as np

# Gradient descent for linear regression
# yhat = wx + b
# loss = (y-yhat)**2 / N

# initialize some parameters
x = np.random.rand(10, 1)
y = 2*x + np.random.rand()
# Parameters
w = .0
b = .0
# Hyperparameter
alpha = .01
# Epochs
epochs = 4000

# Create gradient descent function
def descent(x, y, w, b, alpha):
    dldw = .0
    dldb = .0
    N = x.shape[0]
    # loss = (y - (wx + b))**2
    for xi, yi in zip(x, y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
    
    # Make an update to the w, b parameters
    w = w - alpha*(1/N)*dldw
    b = b - alpha*(1/N)*dldb

    return w, b

# Iteratively make updates
for epoch in range(epochs):
    w, b = descent(x, y, w, b, alpha)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, parameter w: {w}, b: {b}')