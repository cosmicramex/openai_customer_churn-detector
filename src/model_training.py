from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def split_dataset(X, y, test_size=0.2):
    """
    Split dataset into training and testing
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train multiple ML models
    """
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models["Naive Bayes"] = nb

    # SVM
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models
    """
    results = {}
    confusion_matrices = {}

    for name, model in models.items():

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        print("\n========================")
        print(f"Model: {name}")
        print("Accuracy:", acc)
        print(classification_report(y_test, predictions))

        results[name] = {
            "accuracy": acc
        }

        confusion_matrices[name] = confusion_matrix(y_test, predictions)

    return confusion_matrices, results