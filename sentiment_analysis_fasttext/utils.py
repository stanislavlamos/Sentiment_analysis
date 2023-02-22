from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer

stopwords_lst = ENGLISH_STOP_WORDS.union(stopwords.words('english'))


def strip_stopwords(text):
    """
        Removes stop words from a sentence
    """
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_lst]
    return ' '.join(tokens)


def clean_text(text):
    """
        This function strips text of html elements, commas, fullstops brackets and other non-alphanumerics
    """
    soup = BeautifulSoup(text, features="html.parser")
    rgx = r'[^A-Za-z0-9\s\.\']'
    return re.sub(rgx, '', soup.get_text()).replace('.', ' ')


def lemmatize_sentence(text):
    """
        This function lemmatizes a sentence
    """
    lemmatizer = WordNetLemmatizer()

    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess_text(text):
    """
        Preprocesses given dataset text from unnecessary chars
    """
    text = text.lower()
    cleaned_text = clean_text(text)
    cleaned_text = strip_stopwords(cleaned_text)
    cleaned_text = lemmatize_sentence(cleaned_text)

    return cleaned_text


def convert_to_ft_data_format(dataset_name, set_name, x, y):
    """
        Converts data to fasttext input format
    """
    with open(f"./data/{dataset_name}/{dataset_name}.{set_name}", 'w') as write_file:
        y_lst = y.tolist()

        idx = 0
        for _, text in x.items():
            label = y_lst[idx]
            print(f"__label__{label} {text}", file=write_file)
            idx = idx + 1
