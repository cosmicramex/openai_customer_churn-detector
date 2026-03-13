"""
feature_extraction.py

Module responsible for converting processed text into numerical
features using TF-IDF vectorization and N-grams.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features=3000):
    """
    Create a TF-IDF vectorizer with unigram and bigram features.

    Parameters
    ----------
    max_features : int
        Maximum number of features

    Returns
    -------
    TfidfVectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # Unigrams + Bigrams
        lowercase=False       # Already cleaned in preprocessing
    )

    return vectorizer


def fit_vectorizer(vectorizer, train_texts):
    """
    Fit the vectorizer on training data.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
    train_texts : list

    Returns
    -------
    fitted vectorizer
    """

    vectorizer.fit(train_texts)

    print("Vectorizer fitted on training data.")

    return vectorizer


def transform_text(vectorizer, texts):
    """
    Transform text data into TF-IDF feature vectors.

    Parameters
    ----------
    vectorizer : fitted TfidfVectorizer
    texts : list

    Returns
    -------
    sparse matrix
    """

    features = vectorizer.transform(texts)

    print(f"Text transformed into feature matrix: {features.shape}")

    return features


def fit_transform_vectorizer(vectorizer, train_texts):
    """
    Fit vectorizer and transform training text.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
    train_texts : list

    Returns
    -------
    feature_matrix
    """

    features = vectorizer.fit_transform(train_texts)

    print(f"Training feature matrix shape: {features.shape}")

    return features


def get_feature_names(vectorizer):
    """
    Extract vocabulary words used by TF-IDF.

    Useful for visualization of important churn words.

    Returns
    -------
    list
    """

    return vectorizer.get_feature_names_out()


def get_top_words(feature_matrix, feature_names, top_n=20):
    """
    Identify most frequent important words in dataset.

    Parameters
    ----------
    feature_matrix : sparse matrix
    feature_names : list
    top_n : int

    Returns
    -------
    list of (word, score)
    """

    import numpy as np

    word_scores = feature_matrix.sum(axis=0)

    word_scores = np.array(word_scores).flatten()

    sorted_indices = word_scores.argsort()[::-1]

    top_words = []

    for i in sorted_indices[:top_n]:
        top_words.append((feature_names[i], word_scores[i]))

    return top_words