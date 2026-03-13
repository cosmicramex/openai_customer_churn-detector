import matplotlib.pyplot as plt
import seaborn as sns


def plot_churn_distribution(data):

    counts = data["churn"].value_counts()

    plt.figure()
    plt.pie(counts, labels=["Retained", "Churn"], autopct='%1.1f%%')
    plt.title("Customer Churn Distribution")
    plt.show()


def plot_model_accuracy(results):

    names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in names]

    plt.figure()
    plt.bar(names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


def plot_confusion_matrix(cm):

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()