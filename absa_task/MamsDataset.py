import utils
import xml.etree.ElementTree as ET


class Mams:
    def __init__(self):
        self.X_train_full, self.y_train_full = self.preprocess_data("train", "mams_categories")
        self.X_test_full, self.y_test_full = self.preprocess_data("test", "mams_categories")
        self.X_val_full, self.y_val_full = self.preprocess_data("valid", "mams_categories")

        self.polarity_mapping_dict = {
            "positive": 0,
            "negative": 1,
            "neutral": 2
        }

        self.NUM_POLARITIES = 3

        self.labels_mapping_dict = {
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

        self.categories_mapping_dict = {
            "food": 0,
            "service": 1,
            "price": 2,
            "ambience": 3,
            "menu": 4,
            "place": 5,
            "staff": 6,
            "miscellaneous": 7,
        }

        self.NUM_CATEGORIES = 8
        self.NUM_LABELS = 24

    def preprocess_data(self, set_name, dataset_name):
        """
            Function to preprocess training, testing and validation set from MAMS - ACSA dataset
        """
        root = ET.parse(f"./data/{dataset_name}/{set_name}.xml").getroot()

        X = []
        y = []

        for sentence in root.iter('sentence'):
            sentence_text = utils.preprocess_text(sentence[0].text)

            y_tmp = []
            for attributes in sentence[1]:
                attributes = attributes.attrib
                label = attributes['category'] + "_" + attributes['polarity']
                y_tmp.append(label)

            y.append(y_tmp)
            X.append(sentence_text)

        return X, y


    def get_categories_stats(self, y_labels):
        categories_counts = {
            "food": 0,
            "service": 0,
            "price": 0,
            "ambience": 0,
            "menu": 0,
            "place": 0,
            "staff": 0,
            "miscellaneous": 0,
        }
        more_than_one_label = 0

        for idx, sample_lbl in enumerate(y_labels):
            if len(y_labels[idx]) > 1:
                more_than_one_label = more_than_one_label + 1

            for lbl in y_labels[idx]:
                categories_counts[lbl] += 1

        print(f"More than one label: {more_than_one_label / len(y_labels)}")

        return categories_counts

