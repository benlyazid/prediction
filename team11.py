import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random

# Define the lists of values for each feature
cities = ['Fes', 'Casa', 'Rabat']
majors = ['Cs', 'Robotics', 'Medicine', 'Archi']
prices = np.random.randint(100, size=10000)

with open('file.txt', 'r') as f:
    names = f.read().splitlines()
targets = random.choices(names, k=10000)

# Create a dictionary of the data
data = {'city': np.random.choice(cities, size=10000),
        'major': np.random.choice(majors, size=10000),
        'Price': prices,
        'target': targets}

# Create a dataframe from the dictionary
df = pd.DataFrame(data)

# Convert categorical features to numerical using label encoding
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
df['major'] = le.fit_transform(df['major'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Train a KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on the test set
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")