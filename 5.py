import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# Load the dataset
file_path = 'emails.csv'
data_frame = pd.read_csv(file_path)
print("Dataset:\n", data_frame)

# Replace target column values
data_frame['Prediction'] = data_frame['Prediction'].replace({0: 'spam', 1: 'ham'})

# Prepare features and labels
X = data_frame.iloc[:, 1:-1].values
Y = data_frame.iloc[:, -1].values
Y = np.where(Y == 'spam', 1, 0)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=90)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Use only the first two features for 2D plotting
X_train_2D = X_train[:, :2]
X_test_2D = X_test[:, :2]

# Train the SVM model
svm = SVC(kernel='rbf', random_state=0, gamma='scale')
svm.fit(X_train_2D, Y_train)
y_pred_svm = svm.predict(X_test_2D)

print("\n\n\t\t\tClassification Report (SVM)")
print(classification_report(Y_test, y_pred_svm))

# Train the kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)

print("\n\n\t\t\tClassification Report (kNN)")
print(classification_report(Y_test, y_pred_knn))

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.coolwarm)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

# Plot decision boundary for the SVM model
fig, ax = plt.subplots()
plot_decision_boundaries(X_test_2D, Y_test, svm, ax)
plt.title('SVM Decision Boundary with RBF Kernel (2D Data)')
plt.legend()
plt.show()
