import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iterations = 1000
alpha = 0.1

# ML Functions
def compute_cost(X, y, theta, m):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J

def gradient_descent(X, y, z, v, theta, alpha, iterations):
    m = len(y)  # Number of training examples
    n = len(v)  # Number of Validation examples
    cost_history_t = np.zeros(iterations)
    cost_history_v = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta -= sum_delta
        cost_history_t[i] = compute_cost(X, y, theta, m)
        cost_history_v[i] = compute_cost(z, v, theta, n)

    return theta, cost_history_t, cost_history_v



# Use the URL for the raw CSV data
url = 'https://raw.githubusercontent.com/HamedTabkhi/Intro-to-ML/refs/heads/main/Dataset/Housing.csv'

dataset = pd.read_csv(url)
dataset2 = dataset



# Preprocessing for Normalization

from IPython.display import display
# display(dataset)

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
            'airconditioning', 'prefarea']

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

dataset[varlist] = dataset[varlist].apply(binary_map)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
dataset[num_vars] = scaler.fit_transform(dataset[num_vars])



# Part A, Normalization

X_Training = dataset.iloc[dataset.index % 5 != 0, [1, 2, 3, 4, 10]].values
X_Validation = dataset.iloc[dataset.index % 5 == 0, [1, 2, 3, 4, 10]].values
Y_Training = dataset.iloc[dataset.index % 5 != 0, 0].values
Y_Validation = dataset.iloc[dataset.index % 5 == 0, 0].values

# display(X_Training)

t = len(X_Training)
v = len(X_Validation)

X_Training = np.column_stack((np.ones(t, dtype=int), X_Training))
X_Validation = np.column_stack((np.ones(v, dtype=int), X_Validation))

theta = np.zeros(6)

print('Training Set Length: ', t)
print('Validation Set Length: ', v)

theta, cost_history_t, cost_history_v = gradient_descent(X_Training, Y_Training, 
                                                         X_Validation, Y_Validation, 
                                                         theta, alpha, iterations)

plt.plot(range(1, iterations + 1), cost_history_t, color='blue', label = "Training Loss")
plt.plot(range(1, iterations + 1), cost_history_v, color='red', label = "Validation Loss")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.legend()

# Show the plot
plt.show()



# Part B, Normalization

X_Training = dataset.iloc[dataset.index % 5 != 0, 1:12].values
X_Validation = dataset.iloc[dataset.index % 5 == 0, 1:12].values
Y_Training = dataset.iloc[dataset.index % 5 != 0, 0].values
Y_Validation = dataset.iloc[dataset.index % 5 == 0, 0].values

X_Training = np.column_stack((np.ones(t, dtype=int), X_Training))
X_Validation = np.column_stack((np.ones(v, dtype=int), X_Validation))

theta = np.zeros(12)

theta, cost_history_t, cost_history_v = gradient_descent(X_Training, Y_Training, 
                                                         X_Validation, Y_Validation, 
                                                         theta, alpha, iterations)

plt.plot(range(1, iterations + 1), cost_history_t, color='blue', label = "Training Loss")
plt.plot(range(1, iterations + 1), cost_history_v, color='red', label = "Validation Loss")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.legend()

# Show the plot
plt.show()



# Preprocessing for Standardization

dataset = dataset2

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
            'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']

dataset[num_vars] = scaler.fit_transform(dataset[num_vars])

display(dataset)



# Part A, Standardization

X_Training = dataset.iloc[dataset.index % 5 != 0, [1, 2, 3, 4, 10]].values
X_Validation = dataset.iloc[dataset.index % 5 == 0, [1, 2, 3, 4, 10]].values
Y_Training = dataset.iloc[dataset.index % 5 != 0, 0].values
Y_Validation = dataset.iloc[dataset.index % 5 == 0, 0].values

# display(X_Training)

t = len(X_Training)
v = len(X_Validation)

X_Training = np.column_stack((np.ones(t, dtype=int), X_Training))
X_Validation = np.column_stack((np.ones(v, dtype=int), X_Validation))

theta = np.zeros(6)

print('Training Set Length: ', t)
print('Validation Set Length: ', v)

theta, cost_history_t, cost_history_v = gradient_descent(X_Training, Y_Training, 
                                                         X_Validation, Y_Validation, 
                                                         theta, alpha, iterations)

plt.plot(range(1, iterations + 1), cost_history_t, color='blue', label = "Training Loss")
plt.plot(range(1, iterations + 1), cost_history_v, color='red', label = "Validation Loss")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.legend()

# Show the plot
plt.show()



# Part B, Standardization

X_Training = dataset.iloc[dataset.index % 5 != 0, 1:12].values
X_Validation = dataset.iloc[dataset.index % 5 == 0, 1:12].values
Y_Training = dataset.iloc[dataset.index % 5 != 0, 0].values
Y_Validation = dataset.iloc[dataset.index % 5 == 0, 0].values

X_Training = np.column_stack((np.ones(t, dtype=int), X_Training))
X_Validation = np.column_stack((np.ones(v, dtype=int), X_Validation))

theta = np.zeros(12)

theta, cost_history_t, cost_history_v = gradient_descent(X_Training, Y_Training, 
                                                         X_Validation, Y_Validation, 
                                                         theta, alpha, iterations)

plt.plot(range(1, iterations + 1), cost_history_t, color='blue', label = "Training Loss")
plt.plot(range(1, iterations + 1), cost_history_v, color='red', label = "Validation Loss")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.legend()

# Show the plot
plt.show()
