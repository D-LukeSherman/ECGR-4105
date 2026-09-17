import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the URL for the raw CSV data
url = 'https://raw.githubusercontent.com/HamedTabkhi/Intro-to-ML/refs/heads/main/Dataset/Housing.csv'

dataset = pd.read_csv(url)

from IPython.display import display
display(dataset)

X_Training = dataset.iloc[dataset.index % 5 != 0, [1, 2, 3, 4, 10]]
X_Validation = dataset.iloc[dataset.index % 5 == 0, [1, 2, 3, 4, 10]]
Y_Training = dataset.iloc[dataset.index % 5 != 0, [0]]
Y_Validation = dataset.iloc[dataset.index % 5 == 0, [0]]

t = len(X_Training)
v = len(X_Validation)
print('Training Set Length: ', t)
print('Validation Set Length: ', v)
X_Training[0:10]
X_Validation[0:10]

