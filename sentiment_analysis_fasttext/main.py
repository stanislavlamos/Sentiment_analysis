import fasttext
import csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import utils
import itertools


def preprocess_imdb_dataset():
    dataset = pd.read_csv("./data/imdb/data_all_imdb.csv")
    dataset['review'] = dataset['review'].apply(utils.preprocess_text)

    X = dataset['review']
    y = dataset['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)

    utils.imdb_ft_data_format("train", X_train, y_train)
    utils.imdb_ft_data_format("valid", X_val, y_val)

    print("IMDB dataset has been successfully preprocessed")

    return X_test, y_test


def preprocess_sentiment140_dataset():
    train = open('./data/tweets.train', 'w')
    test = open('./data/tweets.valid', 'w')
    with open('./data/training.1600000.processed.noemoticon.csv', mode='r', encoding="ISO-8859-1") as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['target', 'id', 'date', 'flag', 'user', 'text'])
        line = 0
        for row in csv_reader:
            # Clean the training data
            # First we lower case the text
            text = row["text"].lower()
            # remove links
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
            # Remove usernames
            text = re.sub('@[^\s]+', '', text)
            # replace hashtags by just words
            text = re.sub(r'#([^\s]+)', r'\1', text)
            # correct all multiple white spaces to a single white space
            text = re.sub('[\s]+', ' ', text)
            # Additional clean up : removing words less than 3 chars, and remove space at the beginning and teh end
            text = re.sub(r'\W*\b\w{1,3}\b', '', text)
            text = text.strip()
            line = line + 1
            # Split data into train and validation
            if line % 16 == 0:
                print(f'__label__{row["target"]} {text}', file=test)
            else:
                print(f'__label__{row["target"]} {text}', file=train)


def train_fasttext():
    model_tweet = fasttext.train_supervised('./data/tweets.train')
    print(model_tweet.test('./data/tweets.valid'))
    model_tweet.save_model('model.bin')


def ft_classification(dataset_name, x_test):
    print("Starting fasttext model training")

    model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}.train",
                                      autotuneValidationFile=f"./data/{dataset_name}/{dataset_name}.valid")
    samples_number_val, precision_val, recall_val = model.test(f"./data/{dataset_name}/{dataset_name}.valid")
    print(f"Number of validation samples: {samples_number_val}")
    print(f"Precision on validation set: {precision_val}")
    print(f"Recall on validation set: {recall_val}")

    model.save_model(f"./models/ft_classification/{dataset_name}.bin")
    print(f"Model has been saved as: ./models/ft_classification/{dataset_name}.bin")

    y_pred = model.predict(x_test.tolist())
    y_pred_flatten = list(itertools.chain(*y_pred[0]))
    y_pred_flatten_shortened = [label.replace('__label__', '') for label in y_pred_flatten]

    return y_pred_flatten_shortened


def run_on_imdb():
    x_test, y_test = preprocess_imdb_dataset()
    y_pred = ft_classification("imdb", x_test)
    print(classification_report(y_test.tolist(), y_pred))


def run_on_sentiment140():
    pass


if __name__ == '__main__':
    run_on_imdb()
