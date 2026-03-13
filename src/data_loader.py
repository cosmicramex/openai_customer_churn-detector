import pandas as pd


def load_and_prepare_data(file_path):
    """
    Load dataset and convert sentiment to churn labels
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # Convert sentiment to churn label
    # Positive = 0 (not churn)
    # Negative = 1 (churn)

    data["churn"] = data["Sentiment"].apply(
        lambda x: 1 if x.lower() == "negative" else 0
    )

    # Rename review column to feedback
    data = data.rename(columns={"Review_Text": "feedback"})

    return data