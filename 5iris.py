from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Define the split ratios
split_ratios = [(0.9, 0.1), (0.7, 0.3)]

for split_ratio in split_ratios:
    # Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=split_ratio[1], random_state=42)

    # Train and test the Naive Bayes classifier for different k values
    k_values = [3, 5, 7]
    for k in k_values:
        # Train the Naive Bayes classifier on the training data
        model = GaussianNB(var_smoothing=k)
        model.fit(X_train, y_train)

        # Test the Naive Bayes classifier on the testing data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Print the accuracy score for the Iris dataset
        print(f"Iris dataset (k={k}, split={split_ratio}): Accuracy = {accuracy:.2f}")
