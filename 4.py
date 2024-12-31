import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load the dataset
df = pd.read_csv('emails.csv')  # Adjust the path if needed

# Print column names to identify the correct target column
print("Columns in the dataset:", df.columns)

# Use the correct column name for labels
target_column = 'Prediction'

# Check if the target column exists
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in the DataFrame. Available columns are: {df.columns}")

# Convert labels to binary values
df[target_column] = df[target_column].map({'spam': 1, 'ham': 0})

# Drop rows with missing target values
df = df.dropna(subset=[target_column])

# Separate features and target variable
X = df.drop(target_column, axis=1)
y = df[target_column]

# Check for and handle missing values in features
if X.isnull().sum().sum() > 0:
    print("Missing values found in features. Filling with mean values.")
    X = X.fillna(X.mean())

# Print shape of X and y before converting categorical variables
print(f"Shape of X before conversion: {X.shape}")

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Print shape of X after converting categorical variables
print(f"Shape of X after conversion: {X.shape}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the train and test sets
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
