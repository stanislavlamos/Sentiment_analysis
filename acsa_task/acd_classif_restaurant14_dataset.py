import tracemalloc
import utils
import Restaurant2014Dataset
from ft_classif import ft_classification_onelabel, ft_classification_multilabel
from sklearn.metrics import accuracy_score, classification_report
from use_classif import *
import time


def run_onestep_ft():
    """
        Running fasttext multilabel classification on MAMS categories dataset and printing classification report
    """
    my_data = Restaurant2014Dataset.Restaurant2014()
    my_data.create_val_split()
    my_data.y_val_full = utils.delete_polarities(my_data.y_val_full)
    my_data.y_train_full = utils.delete_polarities(my_data.y_train_full)
    my_data.y_test_full = utils.delete_polarities(my_data.y_test_full)

    utils.create_fasttext_input_files("res14", "valid", "ft_acd", my_data.X_val_full,
                                      my_data.y_val_full)
    utils.create_fasttext_input_files("res14", "train", "ft_acd", my_data.X_train_full,
                                      my_data.y_train_full)

    y_pred_labels, y_pred_probs = ft_classification_multilabel("res14",
                                                               my_data.X_test_full,
                                                               train_file_name="ft_acd.train",
                                                               valid_file_name="ft_acd.valid",
                                                               threshold=0.18,
                                                               autotune=False)
    y_test_converted = utils.convert_to_onehot(my_data.y_test_full, my_data.categories_mapping_dict,
                                               labels_count=my_data.NUM_CATEGORIES)
    y_pred_converted = utils.convert_to_onehot(utils.resolve_conflicting_labels(y_pred_labels, y_pred_probs),
                                               my_data.categories_mapping_dict, labels_count=my_data.NUM_CATEGORIES)

    target_names = my_data.categories_mapping_dict.keys()
    print(classification_report(y_test_converted, y_pred_converted, target_names=target_names, zero_division=0))
    print(f"Multi label accuracy (a.k.a. Hamming score): {utils.hamming_score(y_test_converted, y_pred_converted)}")
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_pred_converted)}")


def run_use():
    my_data = Restaurant2014Dataset.Restaurant2014()
    y_preds_onehot = logistic_regression_categories(my_data.X_test_full, my_data.y_test_full, my_data.X_train_full,
                                                    my_data.y_train_full, my_data.categories_mapping_dict, "trans",
                                                    0.35)

    y_test_converted = utils.convert_to_onehot(utils.delete_polarities(my_data.y_test_full), my_data.categories_mapping_dict, my_data.NUM_CATEGORIES)
    print(classification_report(y_test_converted, y_preds_onehot, target_names=my_data.categories_mapping_dict.keys(), zero_division=0))
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_preds_onehot)}")


if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    # run_onestep_ft()
    run_use()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time}")
    current, peak = tracemalloc.get_traced_memory()
    print("Peak was", peak / 10 ** 6, "MB")
    tracemalloc.stop()
