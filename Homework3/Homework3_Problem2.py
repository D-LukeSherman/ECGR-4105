import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

iterations = 1000

# Use the URL for the raw CSV data
url = 'https://raw.githubusercontent.com/D-LukeSherman/ECGR-4105/refs/heads/main/Homework3/Cancer.csv'

dataset = pd.read_csv(url)



# Preprocessing

varlist = ['diagnosis']

def binary_map(x):
    return x.map({'M': 1, 'B': 0})

dataset[varlist] = dataset[varlist].apply(binary_map)

# display(dataset)

# Separating out the features
x = dataset.iloc[:, 2:].values
# Separating out the target
y = dataset.iloc[:, [1]].values

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 0)

t = len(x_train)
v = len(x_valid)
# print(t)
# print(v)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_valid = sc_X.transform(x_valid)



# Logistic Regression

from sklearn.linear_model import SGDClassifier
from sklearn import metrics

model = SGDClassifier(
    loss="log_loss",
    penalty = "l2",
    alpha = 0.01, # Penalty Parameter
    learning_rate="constant",
    eta0=0.01,
    random_state = 0
)

classes = np.unique(y_train)

loss_history_t = []
accuracy_history_t = []

loss_history_v = []
accuracy_history_v = []

for i in range(iterations):

    model.partial_fit(x_train, y_train.ravel(), classes=classes)

    y_train_pred = model.predict(x_train)
    y_train_prob = model.predict_proba(x_train)
    y_valid_pred = model.predict(x_valid)
    y_valid_prob = model.predict_proba(x_valid)

    current_loss_t = metrics.log_loss(y_train, y_train_prob)
    current_accuracy_t = metrics.accuracy_score(y_train, y_train_pred)
    current_loss_v = metrics.log_loss(y_valid, y_valid_prob)
    current_accuracy_v = metrics.accuracy_score(y_valid, y_valid_pred)

    loss_history_t.append(current_loss_t)
    accuracy_history_t.append(current_accuracy_t)
    loss_history_v.append(current_loss_v)
    accuracy_history_v.append(current_accuracy_v)

# print(y_pred[0:10])



# Plots

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_valid, y_valid_pred)
cnf_matrix

#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(y_valid, y_valid_pred))
print("Precision:",metrics.precision_score(y_valid, y_valid_pred))
print("Recall:",metrics.recall_score(y_valid, y_valid_pred))
print("F1:",metrics.f1_score(y_valid, y_valid_pred))

from sklearn.metrics import classification_report
report = classification_report(y_valid, y_valid_pred)
print(report)

import seaborn as sns

class_names = ['B', 'M']  # 0 = Benign, 1 = Malignant

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap
sns.heatmap(
    pd.DataFrame(cnf_matrix),
    annot=True,
    cmap="YlGnBu",
    fmt='g',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)

ax.xaxis.set_label_position("top")

plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()



plt.plot(range(1, iterations + 1), loss_history_t, color='blue', label = "Training Loss")
plt.plot(range(1, iterations + 1), loss_history_v, color='red', label = "Validation Loss")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Show the plot
plt.show()



plt.plot(range(1, iterations + 1), accuracy_history_t, color='blue', label = "Training Accuracy")
plt.plot(range(1, iterations + 1), accuracy_history_v, color='red', label = "Validation Accuracy")
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy History Over Epochs')
plt.legend()

# Show the plot
plt.show()