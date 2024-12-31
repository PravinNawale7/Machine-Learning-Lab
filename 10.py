import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample dataset in the format of transactions
data = {'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'Items': [['I1', 'I2', 'I3'],
                  ['I2', 'I3', 'I4'],
                  ['I4', 'I5'],
                  ['I1', 'I2', 'I4'],
                  ['I1', 'I2', 'I3', 'I5'],
                  ['I1', 'I2', 'I3', 'I4']]}

df = pd.DataFrame(data)

# One hot encoding (each item is represented as a binary value in the transaction)
encoded_data = df['Items'].str.join('|').str.get_dummies()

# Convert the one-hot encoded DataFrame to boolean type to avoid the warning
encoded_data = encoded_data.astype(bool)

# Apply Apriori algorithm with minimum support of 50%
frequent_itemsets = apriori(encoded_data, min_support=0.5, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules with minimum confidence of 70%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the association rules
print("\nAssociation Rules:")
print(rules)
