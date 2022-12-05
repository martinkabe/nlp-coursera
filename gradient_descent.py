# https://www.youtube.com/watch?v=jwStsp8JUPU
import numpy as np
import pandas as pd
from typing import List

rg = np.random.default_rng()

bias = 0.5
alpha = 0.1
epochs = 100

# Load random data
def load_data(n_features, n_values):
    data, weights = generate_data(n_features, n_values)
    features = data.iloc[:,:-1].to_numpy()
    target = np.transpose(data.iloc[:,-1].to_numpy())
    return features, target, weights

# generate dummy data
def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0,1], n_features)
    data = pd.DataFrame(features, columns=["x0","x1","x2"])
    data["targets"] = targets
    return data, weights

# train the model
def train_model(features, weights, bias, alpha, epochs, target, show_epochs=False):
    new_weights = weights
    epoch_loss = []
    for e in range(epochs):
        individual_loss = []
        for feature, t in zip(features, target):
            prediction = get_prediction(feature, new_weights)
            new_weights, bias = gradient_descent(feature, new_weights, t, prediction, alpha, bias)
            individual_loss.append(cross_entropy_loss(t, prediction))
        average_loss = sum(individual_loss)/len(individual_loss)
        if show_epochs:
            print(f'Average LOSS for epoch[{e}] = {average_loss}')
        epoch_loss.append(average_loss)
    return epoch_loss

# Get prediction (model output)
def get_prediction(feature, weights):
    return sigmoid(np.dot(feature, weights) + bias)

# Activation function
def sigmoid(wsum):
    return 1/(1+np.exp(-wsum))

# Loss function
def cross_entropy_loss(target, pred):
    return -(target * np.log10(pred) + (1-target) * (np.log10(1-pred)))

# Gradient descent - update bias and weights
def gradient_descent(feature, weights, target, prediction, alpha, bias):
    # update bias
    bias += alpha * (target - prediction)
    # update weights
    new_w = weights_update = []
    for w, x in zip(weights, feature):
        weights_update.append(w + alpha * (target - prediction) * x)
    # return tupple of updated weights and bias
    return new_w, bias

# Get output - plot of epoch loss and print the original data
def result_output(epoch_loss: List, features, target, weights, print_data: bool = False):
    df = pd.DataFrame(epoch_loss)
    df_plot = df.plot(kind="line", grid=True).get_figure()
    df_plot.savefig("figures/training_loss.pdf")
    if print_data:
        print(f'''
            Features:\n{features}\n
            Target:\n{target}\n
            Weights:\n{weights}\n
            ''')


if __name__=="__main__":
    features, target, weights = load_data(5, 3)
    epoch_loss = train_model(features, weights, bias, alpha, epochs, target)
    result_output(epoch_loss, features, target, weights, True)
