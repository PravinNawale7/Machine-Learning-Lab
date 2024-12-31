import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

df = pd.read_csv('iris_dataset.csv', na_values=['??', '', '/', '###', '$', '&'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # Target (last column)

print(df)
print(X.isnull().sum())

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

print("Missing values after filling:")
print(X_imputed_df.isnull().sum())

print(X_imputed_df)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)

pca = PCA(0.95)
principal_components = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_encoded, cmap='viridis')
plt.title('PCA of IRIS Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()
