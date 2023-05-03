import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Titanic dataset
titanic_df = pd.read_csv('titanicc.csv')

# Preprocess the data (select relevant features and handle missing values)

# Split dataset into features and labels
X = titanic_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
y = titanic_df['Survived']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Handle missing values (e.g., fill with mean)
X = X.fillna(X.mean())

# Split dataset into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate Naive Bayes classifier for different k-values
k_values = [3, 5, 7]
for k in k_values:
    model = GaussianNB(priors=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {accuracy}")
