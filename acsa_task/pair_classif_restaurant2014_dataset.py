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
    utils.create_fasttext_input_files("res14", "valid", "ft_onestep_pair_classif", my_data.X_val_full,
                                      my_data.y_val_full)
    utils.create_fasttext_input_files("res14", "train", "ft_onestep_pair_classif", my_data.X_train_full,
                                      my_data.y_train_full)

    y_pred_labels, y_pred_probs = ft_classification_multilabel("res14",
                                                               my_data.X_test_full,
                                                               train_file_name="ft_onestep_pair_classif.train",
                                                               valid_file_name="ft_onestep_pair_classif.valid",
                                                               threshold=0.18,
                                                               autotune=False)
    y_test_converted = utils.convert_to_onehot(my_data.y_test_full, my_data.labels_mapping_dict,
                                               labels_count=my_data.NUM_LABELS)
    y_pred_converted = utils.convert_to_onehot(utils.resolve_conflicting_labels(y_pred_labels, y_pred_probs),
                                               my_data.labels_mapping_dict, labels_count=my_data.NUM_LABELS)

    target_names = my_data.labels_mapping_dict.keys()
    print(classification_report(y_test_converted, y_pred_converted, target_names=target_names, zero_division=0))
    print(f"Multi label accuracy (a.k.a. Hamming score): {utils.hamming_score(y_test_converted, y_pred_converted)}")
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_pred_converted)}")


def run_twostep_ft():
    """
        Running twostep fasttext classification on MAMS categories dataset and printing classification report
    """
    my_data = Restaurant2014Dataset.Restaurant2014()
    my_data.create_val_split()
    y_test_categories = utils.delete_polarities(my_data.y_test_full)
    y_train_categories = utils.delete_polarities(my_data.y_train_full)
    y_val_categories = utils.delete_polarities(my_data.y_val_full)
    utils.create_fasttext_input_files("res14", "valid", "ft_twotep_pair_classif_categ",
                                      my_data.X_val_full,
                                      y_val_categories)
    utils.create_fasttext_input_files("res14", "train", "ft_twotep_pair_classif_categ",
                                      my_data.X_train_full,
                                      y_train_categories)

    y_pred_categories, y_pred_probs = ft_classification_multilabel("res14",
                                                                   my_data.X_test_full,
                                                                   train_file_name="ft_twotep_pair_classif_categ.train",
                                                                   valid_file_name="ft_twotep_pair_classif_categ.valid",
                                                                   threshold=0.18,
                                                                   autotune=False)

    X_train_polarities, y_train_polarities = utils.create_as_input(my_data.X_train_full, my_data.y_train_full)
    X_val_polarities, y_val_polarities = utils.create_as_input(my_data.X_val_full, my_data.y_val_full)
    utils.create_fasttext_input_files("res14", "valid", "ft_twotep_pair_classif_polar", X_val_polarities,
                                      y_val_polarities)
    utils.create_fasttext_input_files("res14", "train", "ft_twotep_pair_classif_polar", X_train_polarities,
                                      y_train_polarities)
    X_test_polarities = utils.create_as_input_x(my_data.X_test_full, y_pred_categories)

    y_pred_polarities, y_pred_probs = ft_classification_onelabel("res14",
                                                                 X_test_polarities,
                                                                 train_file_name="ft_twotep_pair_classif_polar.train",
                                                                 valid_file_name="ft_twotep_pair_classif_polar.valid",
                                                                 autotune=False)
    y_pred_full = utils.make_full_predictions(y_pred_categories, y_pred_polarities)

    y_test_full_converted = utils.convert_to_onehot(my_data.y_test_full, my_data.labels_mapping_dict,
                                                    my_data.NUM_LABELS)
    y_pred_full_converted = utils.convert_to_onehot(y_pred_full, my_data.labels_mapping_dict,
                                                    my_data.NUM_LABELS)

    target_names = my_data.labels_mapping_dict.keys()
    print(
        classification_report(y_test_full_converted, y_pred_full_converted, target_names=target_names, zero_division=0))
    print(
        f"Multi label accuracy (a.k.a. Hamming score): {utils.hamming_score(y_test_full_converted, y_pred_full_converted)}")
    print(f"Subset accuracy score: {accuracy_score(y_test_full_converted, y_pred_full_converted)}")


def run_use():
    my_data = Restaurant2014Dataset.Restaurant2014()
    y_preds_onehot = logistic_regression_categories(my_data.X_test_full, my_data.y_test_full, my_data.X_train_full,
                                                    my_data.y_train_full, my_data.categories_mapping_dict, "trans",
                                                    0.35)

    y_pred_categories = utils.onehot_to_labels_mapping(y_preds_onehot, my_data.categories_mapping_dict)
    y_full_predicted = logistic_regression_polarities(X_test=my_data.X_test_full,
                                                      y_preds_categories=y_pred_categories,
                                                      y_test_true=my_data.y_test_full,
                                                      X_train=my_data.X_train_full,
                                                      y_train=my_data.y_train_full,
                                                      mapping_dict_categories=my_data.categories_mapping_dict,
                                                      mapping_dict_polarities=my_data.polarity_mapping_dict,
                                                      encoder_type="trans"
                                                      )

    y_test_converted = utils.convert_to_onehot(my_data.y_test_full, my_data.labels_mapping_dict, my_data.NUM_LABELS)
    y_pred_converted = utils.convert_to_onehot(y_full_predicted, my_data.labels_mapping_dict, my_data.NUM_LABELS)
    print(classification_report(y_test_converted, y_pred_converted, target_names=my_data.labels_mapping_dict.keys(), zero_division=0))
    print(f"Subset accuracy score: {accuracy_score(y_test_converted, y_pred_converted)}")


if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    # run_onestep_ft()
    # run_twostep_ft()
    run_use()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time}")
    current, peak = tracemalloc.get_traced_memory()
    print("Peak was", peak / 10 ** 6, "MB")
    tracemalloc.stop()
