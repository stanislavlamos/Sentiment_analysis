import fasttext
from sklearn.metrics import classification_report
import numpy as np
import utils
import xml.etree.ElementTree as ET
from collections import Counter


labels_mapping_dict = {
    "food_positive": 0,
    "food_negative": 1,
    "food_neutral": 2,
    "service_positive": 3,
    "service_negative": 4,
    "service_neutral": 5,
    "price_positive": 6,
    "price_negative": 7,
    "price_neutral": 8,
    "ambience_positive": 9,
    "ambience_negative": 10,
    "ambience_neutral": 11,
    "menu_positive": 12,
    "menu_negative": 13,
    "menu_neutral": 14,
    "place_positive": 15,
    "place_negative": 16,
    "place_neutral": 17,
    "staff_positive": 18,
    "staff_negative": 19,
    "staff_neutral": 20,
    "miscellaneous_positive": 21,
    "miscellaneous_negative": 22,
    "miscellaneous_neutral": 23
}

NUM_LABELS = 24


def preprocess_train_val_set(set_name, dataset_name):
    """
        Function to preprocess training and validation set and create training and validation file for fasttext
    """
    print(f"Preprocessing of {set_name} set from {dataset_name} dataset...")

    root = ET.parse(f"./data/{dataset_name}/{set_name}.xml").getroot()

    with open(f"./data/{dataset_name}/{dataset_name}.{set_name}", 'w') as write_file:
        for sentence in root.iter('sentence'):
            sentence_text = utils.preprocess_text(sentence[0].text)

            labels = ""
            for attributes in sentence[1]:
                attributes = attributes.attrib
                label = "__label__" + attributes['category'] + "_" + attributes['polarity']
                labels = labels + label + " "

            print(labels + sentence_text, file=write_file)

    print(f"Preprocessing has successfully ended, {set_name} file created")


def preprocess_test_set(dataset_name):
    """
        Function to preprocess test set and return X_test and y_test for multilabel fasttext
    """
    print(f"Preprocessing of test set from {dataset_name} dataset...")

    root = ET.parse(f"./data/{dataset_name}/test.xml").getroot()

    X_test = []
    y_test = []

    for sentence in root.iter('sentence'):
        sentence_text = utils.preprocess_text(sentence[0].text)
        X_test.append(sentence_text)

        labels = []
        for attributes in sentence[1]:
            attributes = attributes.attrib
            label = attributes['category'] + "_" + attributes['polarity']
            labels.append(label)
        y_test.append(labels)

    print(f"Preprocessing has successfully ended, test file created")

    return X_test, y_test


def resolve_conflicting_labels(y_pred_labels, y_pred_probs):
    """
        Function to make sure that for each classified label we get only most probable polarity label
    """
    nonconflicting_labels_lst = []

    for outer_labels, outer_probs in zip(y_pred_labels, y_pred_probs):
        categories_lst = [lbl.split('_')[0] for lbl in outer_labels]
        category_occurencies = Counter(categories_lst)

        conflicting_categories = [key for key, value in category_occurencies.items() if value > 1]
        polarities = ["positive", "negative", "neutral"]
        probs = [-1, -1, -1]
        best_duplicate_labels = []

        for conflict_category in conflicting_categories:
            if conflict_category + "_" + "negative" in outer_labels:
                probs[1] = (outer_probs[outer_labels.index(conflict_category + "_" + "negative")])
            if conflict_category + "_" + "positive" in outer_labels:
                probs[0] = (outer_probs[outer_labels.index(conflict_category + "_" + "positive")])
            if conflict_category + "_" + "neutral" in outer_labels:
                probs[2] = (outer_probs[outer_labels.index(conflict_category + "_" + "neutral")])

            argmax_polarity = polarities[utils.argmax(probs)]
            best_duplicate_labels.append(conflict_category + "_" + argmax_polarity)

        cleared_conflicted_labels = []
        for outer_label in outer_labels:
            outer_label_new = outer_label.split('_')[0]

            if outer_label_new in conflicting_categories and (outer_label not in best_duplicate_labels):
                continue

            cleared_conflicted_labels.append(outer_label)

        nonconflicting_labels_lst.append(cleared_conflicted_labels)

    return nonconflicting_labels_lst


def ft_classification(dataset_name, x_test):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}.train", loss='ova', lr=0.05,
                                      epoch=1000, dim=300)

    # print("Generated hyperparameters")
    # args_obj = model.f.getArgs()
    # for hparam in dir(args_obj):
    #     if not hparam.startswith('__'):
    #         print(f"{hparam} -> {getattr(args_obj, hparam)}")

    samples_number_val, precision_val, recall_val = model.test(f"./data/{dataset_name}/{dataset_name}.valid")
    print(f"Number of validation samples: {samples_number_val}")
    print(f"Precision on validation set: {precision_val}")
    print(f"Recall on validation set: {recall_val}")

    y_pred = model.predict(x_test, k=-1, threshold=0.8)
    y_pred_probs = y_pred[1]

    y_pred_shortened = []
    for possible_labels in y_pred[0]:
        y_pred_shortened_inner = []
        for cur_label in possible_labels:
            cur_label_cleared = cur_label.replace('__label__', '')
            y_pred_shortened_inner.append(cur_label_cleared)
        y_pred_shortened.append(y_pred_shortened_inner)

    return y_pred_shortened, y_pred_probs


def convert_to_multiclass_report_format(input_labels):
    """
        Converting resulting multilabel classification to sklearn.metrics.classification_report input format
    """
    output_lst = np.zeros((len(input_labels), NUM_LABELS))

    for outer_idx, text_labels_arr in enumerate(input_labels):
        for individual_label in text_labels_arr:
            mapping_idx = labels_mapping_dict[individual_label]
            output_lst[outer_idx][mapping_idx] = 1

    return output_lst.tolist()


def run_on_mams_categories():
    """
        Running fasttext multilabel classification on MAMS categories dataset and printing classification report
    """
    preprocess_train_val_set("train", "mams_categories")
    preprocess_train_val_set("valid", "mams_categories")
    X_test, y_test = preprocess_test_set("mams_categories")

    y_pred_labels, y_pred_probs = ft_classification("mams_categories", X_test)
    y_test_converted = convert_to_multiclass_report_format(y_test)
    y_pred_converted = convert_to_multiclass_report_format(resolve_conflicting_labels(y_pred_labels, y_pred_probs))

    target_names = labels_mapping_dict.keys()
    print(classification_report(y_test_converted, y_pred_converted, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    run_on_mams_categories()
