from src.data_loader import load_and_prepare_data
from src.text_preprocessing import preprocess_text
from src.feature_extraction import create_vectorizer, fit_transform_vectorizer
from src.model_training import split_dataset, train_models, evaluate_models
from src.prediction import predict_feedback
from src.visualization import plot_churn_distribution, plot_model_accuracy, plot_confusion_matrix


# 1 Load dataset
data = load_and_prepare_data("dataset/chatgpt_reviews_1050_dataset_final.csv")


# 2 Preprocess text
data["clean_text"] = data["feedback"].apply(preprocess_text)


# 3 Feature extraction
vectorizer = create_vectorizer()

X = fit_transform_vectorizer(vectorizer, data["clean_text"])
y = data["churn"]


# 4 Train test split
X_train, X_test, y_train, y_test = split_dataset(X, y)


# 5 Train models
models = train_models(X_train, y_train)


# 6 Evaluate models
confusion_matrices, results = evaluate_models(models, X_test, y_test)

print("\nFinal Model Results\n")

for model, metrics in results.items():
    print(model)
    print(metrics)
    print()


# 7 Visualization
plot_churn_distribution(data)
plot_model_accuracy(results)


# 8 Select best model
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = models[best_model_name]

print("Best Model:", best_model_name)


# 9 Confusion matrix
plot_confusion_matrix(confusion_matrices[best_model_name])


# 10 Test prediction
sample_text = "The responses are slow and inaccurate"

prediction = predict_feedback(sample_text, vectorizer, best_model)

print("\nPrediction:", prediction)