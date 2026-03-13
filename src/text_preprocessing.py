import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)