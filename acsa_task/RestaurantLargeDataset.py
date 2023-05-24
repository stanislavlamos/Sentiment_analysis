import utils
import json
from sklearn.model_selection import train_test_split
import itertools


class RestaurantLarge:
    def __init__(self):
        self.X_train_full, self.y_train_full = self.preprocess_data("train")
        self.X_test_full, self.y_test_full = self.preprocess_data("test")
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
            "drinks": 4,
            "restaurant": 5,
            "location": 6,
            "misc": 7
        }

        self.NUM_CATEGORIES = 8

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
            "location_positive": 9,
            "location_negative": 10,
            "location_neutral": 11,
            "service_positive": 12,
            "service_negative": 13,
            "service_neutral": 14,
            "restaurant_positive": 15,
            "restaurant_negative": 16,
            "restaurant_neutral": 17,
            "drinks_positive": 18,
            "drinks_negative": 19,
            "drinks_neutral": 20,
            "misc_positive": 21,
            "misc_negative": 22,
            "misc_neutral": 23
        }

        self.NUM_LABELS = 24

    def preprocess_data(self, set_name):
        X = []
        y = []
        tmp_data_holder = []

        with open(f"./data/restaurant_large/acsa_{set_name}.json") as input_file:
            json_file = json.load(input_file)
            for record in json_file:
                cur_sentence = record["sentence"]
                cur_category = record["aspect"]
                cur_polarity = record["sentiment"]
                tmp_data_holder.append([cur_sentence, cur_category + "_" + cur_polarity])

        merged_data_holder = [[key, [n for _, n in grp]] for key, grp in itertools.groupby(tmp_data_holder, key=lambda x: x[0])]

        for inner_lst in merged_data_holder:
            X.append(utils.preprocess_text(inner_lst[0]))
            y.append(inner_lst[1])

        return X, y

    def create_val_split(self):
        self.X_train_full, self.X_val_full, self.y_train_full, self.y_val_full = train_test_split(
            self.X_train_full,
            self.y_train_full,
            test_size=0.12,
            shuffle=True,
            random_state=1)


    def get_categories_stats(self, y_labels):
        categories_counts = {
            "food": 0,
            "service": 0,
            "price": 0,
            "ambience": 0,
            "drinks": 0,
            "restaurant": 0,
            "location": 0,
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

