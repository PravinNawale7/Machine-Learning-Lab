import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('country.csv', na_values=['&', '?', '/', '#', '$'])

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Income'] = df['Income'].fillna(df['Income'].mean())
df['Region'] = df['Region'].fillna(df['Region'].value_counts().idxmax())

print("Counts of each region after filling missing values:")
print(df['Region'].value_counts())
print()

label_encoder = LabelEncoder()
df['Region_Label_Encoding'] = label_encoder.fit_transform(df['Region'])

print("DataFrame after encoding categorical data:")
print(df)
print()

scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

print("DataFrame after scaling 'Age' and 'Income':")
print(df)
print()

X = df.drop(['Region', 'Region_Label_Encoding'], axis=1)
y = df['Region_Label_Encoding']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print()
