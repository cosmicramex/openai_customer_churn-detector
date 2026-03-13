def predict_feedback(model, vectorizer, preprocess_function, text):

    clean_text = preprocess_function(text)

    vector = vectorizer.transform([clean_text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0][1]

    if prediction == 1:
        label = "Churn"
    else:
        label = "Retained"

    return label, probability