import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('collegePlace.csv')

# Preprocess the data
x = df.drop(['PlacedOrNot', 'Age', 'Hostel'], axis=1)
y = df['PlacedOrNot']
le = preprocessing.LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
x['Stream'] = le.fit_transform(x['Stream'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Train a RandomForestClassifier model
classify = RandomForestClassifier(n_estimators=10, criterion="entropy")
classify.fit(x_train, y_train)

# Save the trained model
pickle.dump(classify, open('model.pkl', 'wb'))

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

