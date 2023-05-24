from ft_classif import ft_classification_onelabel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import utils
import time
import json
from lgr_classif import lgr_binary_classif
import tracemalloc


def preprocess_imdb_dataset():
    """
        Preprocessing of IMDB dataset and returning x and y from test set
    """
    print("Preprocessing IMDB dataset...")

    #dataset = pd.read_csv("./data/imdb/data_all_imdb.csv")
    #dataset['review'] = dataset['review'].apply(utils.preprocess_text)
#
    #X = dataset['review']
    #y = dataset['sentiment']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []

    with open("./data/imdb/test.jsonl", 'r') as test_file:
        for line in test_file:
            line_dict = json.loads(line)
            line_text = utils.preprocess_text(line_dict['text'])
            line_label = line_dict['label']

            if line_label == 0:
                line_label = "negative"
            else:
                line_label = "positive"

            X_test.append(line_text)
            y_test.append(line_label)

    with open("./data/imdb/train.jsonl", 'r') as train_file:
        for line in train_file:
            line_dict = json.loads(line)
            line_text = utils.preprocess_text(line_dict['text'])
            line_label = line_dict['label']

            if line_label == 0:
                line_label = "negative"
            else:
                line_label = "positive"

            X_train.append(line_text)
            y_train.append(line_label)

    with open("./data/imdb/dev.jsonl", 'r') as dev_file:
        for line in dev_file:
            line_dict = json.loads(line)
            line_text = utils.preprocess_text(line_dict['text'])
            line_label = line_dict['label']

            if line_label == 0:
                line_label = "negative"
            else:
                line_label = "positive"

            X_val.append(line_text)
            y_val.append(line_label)

    # utils.apply_vocab_length_to_series(X, "all", "IMDB")
    # utils.apply_vocab_length_to_series(X_train, "train", "IMDB")
    # utils.apply_vocab_length_to_series(X_val, "validation", "IMDB")
    # utils.apply_vocab_length_to_series(X_test, "test", "IMDB")

    print("IMDB dataset has been successfully preprocessed")

    return X_test, y_test, X_train, y_train, X_val, y_val


def preprocess_sentiment140_dataset():
    """
        Preprocessing of Sentiment140 dataset and returning x and y from test set
    """
    print("Preprocessing Sentiment140 dataset...")

    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    dataset = pd.read_csv("./data/sentiment140/data_all_sentiment140.csv", encoding='latin-1', names=cols)
    dataset['text'] = dataset['text'].apply(utils.preprocess_text)

    X = dataset['text']
    y = dataset['sentiment'].replace({0: "negative", 2: "neutral", 4: "positive"})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1, shuffle=True)

    # utils.apply_vocab_length_to_series(X, "all", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_train, "train", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_val, "validation", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_test, "test", "Sentiment140")

    print("Sentiment140 dataset has been successfully preprocessed")

    return X_test, y_test, X_train, y_train, X_val, y_val


def preprocess_rotten_tomatoes_dataset():
    """
        Preprocessing of reviews from rotten tomatoes dataset
    """
    print("Preprocessing Rotten Tomatoes dataset...")

    cols = ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
    dataset = pd.read_csv("./data/rotten_tomatoes/train.tsv", sep='\t', names=cols)
    dataset['Phrase'] = dataset['Phrase'].apply(utils.preprocess_text)

    X = dataset['Phrase']
    y = dataset['Sentiment']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

    cols_test = ['PhraseId', 'SentenceId', 'Phrase']
    dataset_test = pd.read_csv("./data/rotten_tomatoes/test.tsv", sep='\t', names=cols_test)
    dataset_test['Phrase'] = dataset['Phrase'].apply(utils.preprocess_text)
    X_test = dataset_test['Phrase']
    ids = dataset_test['PhraseId']

    utils.convert_to_ft_data_format("rotten_tomatoes", "train", X_train, y_train)
    utils.convert_to_ft_data_format("rotten_tomatoes", "valid", X_val, y_val)

    print("Rotten tomatoes dataset has been successfully preprocessed")

    return X_test, ids


def run_ft_on_imdb():
    """
        Running fasttext classification on IMDB dataset and printing classification report
    """
    X_test, y_test, X_train, y_train, X_val, y_val = preprocess_imdb_dataset()
    utils.create_fasttext_input_files_binary("imdb", "train", "imdb", X_train, y_train)
    utils.create_fasttext_input_files_binary("imdb", "valid", "imdb", X_val, y_val)

    y_pred, _ = ft_classification_onelabel("imdb", X_test, autotune=False)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")


def run_lgr_on_imdb():
    X_test, y_test, X_train, y_train, X_val, y_val = preprocess_imdb_dataset()
    y_pred = lgr_binary_classif(X_train, y_train, X_test, "dan")

    print(classification_report(y_test, y_pred))
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")


def run_ft_on_sentiment140():
    """
        Running fasttext classification on Sentiment140 dataset and printing classification report
    """
    X_test, y_test, X_train, y_train, X_val, y_val = preprocess_sentiment140_dataset()
    utils.create_fasttext_input_files_binary("sentiment140", "train", "sentiment140", X_train.tolist(), y_train.tolist())
    utils.create_fasttext_input_files_binary("sentiment140", "valid", "sentiment140", X_val.tolist(), y_val.tolist())

    y_pred, _ = ft_classification_onelabel("sentiment140", X_test.tolist(), autotune=False)
    print(classification_report(y_test.tolist(), y_pred))
    print(f"Accuracy score: {accuracy_score(y_test.tolist(), y_pred)}")


def run_lgr_sentiment140():
    X_test, y_test, X_train, y_train, X_val, y_val = preprocess_sentiment140_dataset()
    y_pred = lgr_binary_classif(X_train, y_train, X_test, "trans")

    print(classification_report(y_test, y_pred))
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")


def run_on_ft_rotten_tomatoes():
    """
        Running fasttext classification on Rotten tomatoes dataset and printing classification report
    """
    x_test, ids = preprocess_rotten_tomatoes_dataset()
    y_pred = ft_classification_onelabel("rotten_tomatoes", x_test, autotune=True)

    df = pd.DataFrame({"PhraseId": ids.tolist(), "Sentiment": y_pred})
    df.to_csv("./data/rotten_tomatoes/kaggle_submission.csv", index=False)


if __name__ == '__main__':
    start_time = time.time()
    tracemalloc.start()
    run_ft_on_sentiment140()
    # run_lgr_sentiment140()
    # run_ft_on_imdb()
    # run_lgr_on_imdb()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time}")
    current, peak = tracemalloc.get_traced_memory()
    print("Peak was", peak / 10 ** 6, "MB")
    tracemalloc.stop()
