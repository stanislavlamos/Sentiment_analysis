from sklearn.model_selection import train_test_split
import utils
import xml.etree.ElementTree as ET


class Restaurant2014:
    def __init__(self):
        self.X_train_full, self.y_train_full = self.preprocess_data("train", "res14")
        self.X_test_full, self.y_test_full = self.preprocess_data("test", "res14")
        self.X_val_full = None
        self.y_val_full = None

        self.polarity_mapping_dict = {
            "positive": 0,
            "negative": 1,
            "neutral": 2
        }

        self.NUM_POLARITIES = 3

        self.categories_mapping_dict = {
            "food": 0,
            "service": 1,
            "price": 2,
            "ambience": 3,
            "misc": 4
        }

        self.NUM_CATEGORIES = 5

        self.labels_mapping_dict = {
            "food_positive": 0,
            "food_negative": 1,
            "food_neutral": 2,
            "price_positive": 3,
            "price_negative": 4,
            "price_neutral": 5,
            "ambience_positive": 6,
            "ambience_negative": 7,
            "ambience_neutral": 8,
            "service_positive": 9,
            "service_negative": 10,
            "service_neutral": 11,
            "misc_positive": 12,
            "misc_negative": 13,
            "misc_neutral": 14
        }

        self.NUM_LABELS = 15

    def preprocess_data(self, set_name, dataset_name):
        root = ET.parse(f"./data/{dataset_name}/Restaurants_{set_name}.xml").getroot()

        X = []
        y = []

        for sentence in root.iter('sentence'):
            sentence_text = utils.preprocess_text(sentence[0].text)

            categories_tag_idx = 2
            if sentence[1].tag == "aspectCategories":
                categories_tag_idx = 1

            if len(sentence[categories_tag_idx]) == 1 and sentence[categories_tag_idx][0].attrib[
                "polarity"] == "conflict":
                continue

            y_tmp = []
            for attributes in sentence[categories_tag_idx]:
                attributes = attributes.attrib

                cur_polarity = attributes['polarity']
                if cur_polarity == "conflict":
                    continue

                cur_category = attributes['category']
                if cur_category == "anecdotes/miscellaneous":
                    cur_category = "misc"

                label = cur_category + "_" + cur_polarity
                y_tmp.append(label)

            y.append(y_tmp)
            X.append(sentence_text)

        return X, y

    def create_val_split(self):
        self.X_train_full, self.X_val_full, self.y_train_full, self.y_val_full = train_test_split(
            self.X_train_full,
            self.y_train_full,
            test_size=0.15,
            shuffle=True,
            random_state=1)


    def get_categories_stats(self, y_labels):
        categories_counts = {
            "food": 0,
            "service": 0,
            "price": 0,
            "ambience": 0,
            "misc": 0
        }
        more_than_one_label = 0

        for idx, sample_lbl in enumerate(y_labels):
            if len(y_labels[idx]) > 1:
                more_than_one_label = more_than_one_label + 1

            for lbl in y_labels[idx]:
                categories_counts[lbl] += 1

        print(f"More than one label: {more_than_one_label / len(y_labels)}")

        return categories_counts
