import MamsDataset
import utils
from sklearn.metrics import classification_report, accuracy_score
from ft_classif import ft_classification_multilabel
from lstm_classif import tokenize_sentences, train_lstm_multilabel, evaluate_lstm
from use_classif import *
import time
import tracemalloc


def run_ft():
    my_data = MamsDataset.Mams()
    my_data.y_val_full = utils.delete_polarities(my_data.y_val_full)
    my_data.y_train_full = utils.delete_polarities(my_data.y_train_full)
    my_data.y_test_full = utils.delete_polarities(my_data.y_test_full)

    utils.create_fasttext_input_files("mams_categories", "valid", "ft_acd", my_data.X_val_full,
                                      my_data.y_val_full)
    utils.create_fasttext_input_files("mams_categories", "train", "ft_acd", my_data.X_train_full,
                                      my_data.y_train_full)

    y_pred_labels, y_pred_probs = ft_classification_multilabel("mams_categories",
                                                               my_data.X_test_full,
                                                               train_file_name="ft_acd.train",
                                                               valid_file_name="ft_acd.valid",
                                                               threshold=0.5,
                                                               autotune=False)
    y_test_converted = utils.convert_to_onehot(my_data.y_test_full, my_data.categories_mapping_dict,
                                               labels_count=my_data.NUM_CATEGORIES)
    y_pred_converted = utils.convert_to_onehot(utils.resolve_conflicting_labels(y_pred_labels, y_pred_probs),
                                               my_data.categories_mapping_dict, labels_count=my_data.NUM_CATEGORIES)

    target_names = my_data.categories_mapping_dict.keys()
    print(classification_report(y_test_converted, y_pred_converted, target_names=target_names, zero_division=0))
    print(f"Multi label accuracy (a.k.a. Hamming score): {utils.hamming_score(y_test_converted, y_pred_converted)}")
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_pred_converted)}")


def run_lstm():
    my_data = MamsDataset.Mams()
    my_data.y_val_full = utils.delete_polarities(my_data.y_val_full)
    my_data.y_train_full = utils.delete_polarities(my_data.y_train_full)
    my_data.y_test_full = utils.delete_polarities(my_data.y_test_full)

    y_train = utils.convert_to_onehot(my_data.y_train_full, my_data.categories_mapping_dict, my_data.NUM_CATEGORIES)
    y_test = utils.convert_to_onehot(my_data.y_test_full, my_data.categories_mapping_dict, my_data.NUM_CATEGORIES)
    y_val = utils.convert_to_onehot(my_data.y_val_full, my_data.categories_mapping_dict, my_data.NUM_CATEGORIES)

    X_train, vocab_size = tokenize_sentences(my_data.X_train_full)
    X_test, _ = tokenize_sentences(my_data.X_test_full)
    X_val, _ = tokenize_sentences(my_data.X_val_full)

    model = train_lstm_multilabel(vocab_size, my_data.NUM_CATEGORIES, X_train, y_train, X_val, y_val)
    evaluate_lstm(model, X_test, y_test, 0.1, my_data.categories_mapping_dict)


def run_use():
    my_data = MamsDataset.Mams()
    y_pred_categories_onehot = logistic_regression_categories(my_data.X_test_full, my_data.y_test_full,
                                                              my_data.X_train_full,
                                                              my_data.y_train_full, my_data.categories_mapping_dict,
                                                              "trans",
                                                              0.38)

    y_test_converted = utils.convert_to_onehot(utils.delete_polarities(my_data.y_test_full),
                                               my_data.categories_mapping_dict,
                                               my_data.NUM_CATEGORIES)

    print(
        classification_report(y_test_converted, y_pred_categories_onehot,
                              target_names=my_data.categories_mapping_dict.keys()))
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_pred_categories_onehot)}")


if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    # run_ft()
    run_use()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time}")
    current, peak = tracemalloc.get_traced_memory()
    print("Peak was", peak / 10 ** 6, "MB")
    tracemalloc.stop()

