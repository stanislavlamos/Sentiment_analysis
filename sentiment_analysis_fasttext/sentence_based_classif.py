import fasttext
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import utils
import itertools


def preprocess_imdb_dataset():
    """
        Preprocessing of IMDB dataset and returning x and y from test set
    """
    print("Preprocessing IMDB dataset...")

    dataset = pd.read_csv("./data/imdb/data_all_imdb.csv")
    dataset['review'] = dataset['review'].apply(utils.preprocess_text)

    X = dataset['review']
    y = dataset['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)

    # utils.apply_vocab_length_to_series(X, "all", "IMDB")
    # utils.apply_vocab_length_to_series(X_train, "train", "IMDB")
    # utils.apply_vocab_length_to_series(X_val, "validation", "IMDB")
    # utils.apply_vocab_length_to_series(X_test, "test", "IMDB")

    utils.convert_to_ft_data_format("imdb", "train", X_train, y_train)
    utils.convert_to_ft_data_format("imdb", "valid", X_val, y_val)

    print("IMDB dataset has been successfully preprocessed")

    return X_test, y_test


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)

    # utils.apply_vocab_length_to_series(X, "all", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_train, "train", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_val, "validation", "Sentiment140")
    # utils.apply_vocab_length_to_series(X_test, "test", "Sentiment140")

    utils.convert_to_ft_data_format("sentiment140", "train", X_train, y_train)
    utils.convert_to_ft_data_format("sentiment140", "valid", X_val, y_val)

    print("Sentiment140 dataset has been successfully preprocessed")

    return X_test, y_test


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


def ft_classification(dataset_name, x_test):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    # To successfully run on Rotten Tomatoes dataset add parameter  -autotuneModelSize="2G"
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
    """
        Running fasttext classification on IMDB dataset and printing classification report
    """
    x_test, y_test = preprocess_imdb_dataset()
    y_pred = ft_classification("imdb", x_test)
    print(classification_report(y_test.tolist(), y_pred))
    print(f"Accuracy score: {accuracy_score(y_test.tolist(), y_pred)}")


def run_on_sentiment140():
    """
        Running fasttext classification on Sentiment140 dataset and printing classification report
    """
    x_test, y_test = preprocess_sentiment140_dataset()
    y_pred = ft_classification("sentiment140", x_test)
    print(classification_report(y_test.tolist(), y_pred))
    print(f"Accuracy score: {accuracy_score(y_test.tolist(), y_pred)}")


def run_on_rotten_tomatoes():
    """
        Running fasttext classification on Rotten tomatoes dataset and printing classification report
    """
    x_test, ids = preprocess_rotten_tomatoes_dataset()
    y_pred = ft_classification("rotten_tomatoes", x_test)

    df = pd.DataFrame({"PhraseId": ids.tolist(), "Sentiment": y_pred})
    df.to_csv("./data/rotten_tomatoes/kaggle_submission.csv", index=False)


if __name__ == '__main__':
    # run_on_imdb()
    run_on_sentiment140()
    # run_on_rotten_tomatoes()
