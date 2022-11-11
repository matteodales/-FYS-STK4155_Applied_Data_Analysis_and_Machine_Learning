
# Cost function gradients used in the training of the neural network

# regression
def regr_cost_grad(y_data, y_tilde):
    return (2 / y_tilde.shape[0]) * (y_tilde - y_data)

# classification
def class_cost_grad(y_data, y_tilde):
        return (y_tilde - y_data) / y_tilde.shape[0]