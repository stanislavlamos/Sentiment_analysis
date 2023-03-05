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


def apply_vocab_length_to_series(x, set_name, dataset_name):
    """
        Get vocab size from words in each sentence of the pandas series
    """
    vocab = []

    for _, text in x.items():
        vocab = get_vocabulary_length(text, vocab)

    word_len_avg = sum(map(len, vocab)) / len(vocab)
    print(f"Vocabulary length of {set_name} set in {dataset_name} dataset: {len(vocab)}")
    print(f"Average word length of {set_name} set in {dataset_name} dataset: {word_len_avg}")


def get_vocabulary_length(sentence, vocab):
    """
        Update vocabulary with words from input sentence
    """
    words = sentence.split()

    for word in words:
        if word not in vocab:
            vocab.append(word)

    return vocab


def argmax(lst):
    """
        Get index of the maximum element from list
    """
    return lst.index(max(lst))
