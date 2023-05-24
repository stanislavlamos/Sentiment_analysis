from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter


stopwords_lst = ENGLISH_STOP_WORDS.union(stopwords.words('english'))


def strip_stopwords(text):
    """
        Removes stop words from a sentence
    """
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_lst]
    return ' '.join(tokens)


def clean_text(text):
    """
        This function strips text of html elements, commas, fullstops brackets and other non-alphanumerics
    """
    soup = BeautifulSoup(text, features="html.parser")
    rgx = r'[^A-Za-z0-9\s\.\']'
    return re.sub(rgx, '', soup.get_text()).replace('.', ' ')


def lemmatize_sentence(text):
    """
        This function lemmatizes a sentence
    """
    lemmatizer = WordNetLemmatizer()

    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess_text(text):
    """
        Preprocesses given dataset text from unnecessary chars
    """
    text = text.lower()
    text = text.strip()
    cleaned_text = re.sub(' +', ' ', clean_text(text))
    cleaned_text = strip_stopwords(cleaned_text)
    cleaned_text = lemmatize_sentence(cleaned_text)

    return cleaned_text


def convert_to_ft_data_format(dataset_name, set_name, x, y):
    """
        Converts data to fasttext input format
    """
    with open(f"./data/{dataset_name}/{dataset_name}.{set_name}", 'w') as write_file:
        y_lst = y.tolist()

        idx = 0
        for _, text in x.items():
            label = y_lst[idx]
            print(f"__label__{label} {text}", file=write_file)
            idx = idx + 1


def apply_vocab_length_to_series(x, set_name, dataset_name):
    """
        Get vocab size from words in each sentence of the pandas series
    """
    vocab = []

    for _, text in x.items():
        vocab = get_vocabulary_length(text, vocab)

    word_len_avg = sum(map(len, vocab)) / len(vocab)
    print(f"Vocabulary length of {set_name} set in {dataset_name} dataset: {len(vocab)}")
    print(f"Average word length of {set_name} set in {dataset_name} dataset: {word_len_avg}")


def get_vocabulary_length(sentence, vocab):
    """
        Update vocabulary with words from input sentence
    """
    words = sentence.split()

    for word in words:
        if word not in vocab:
            vocab.append(word)

    return vocab


def argmax(lst):
    """
        Get index of the maximum element from list
    """
    return lst.index(max(lst))


def convert_to_onehot(input_labels, mapping_dict, labels_count=24):
    output_lst = np.zeros((len(input_labels), labels_count))

    for outer_idx, text_labels_arr in enumerate(input_labels):
        for individual_label in text_labels_arr:
            mapping_idx = mapping_dict[individual_label]
            output_lst[outer_idx][mapping_idx] = 1

    return output_lst.tolist()


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

            argmax_polarity = polarities[argmax(probs)]
            best_duplicate_labels.append(conflict_category + "_" + argmax_polarity)

        cleared_conflicted_labels = []
        for outer_label in outer_labels:
            outer_label_new = outer_label.split('_')[0]

            if outer_label_new in conflicting_categories and (outer_label not in best_duplicate_labels):
                continue

            cleared_conflicted_labels.append(outer_label)

        nonconflicting_labels_lst.append(cleared_conflicted_labels)

    return nonconflicting_labels_lst


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def filter_onelabel_sentences(X, y):
    new_X = []
    new_y = []
    for i in range(len(y)):
        if len(y[i]) == 1:
            new_y.append(y[i])
            new_X.append(X[i])

    return new_X, new_y


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
        Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label classification
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def delete_polarities(y):
    y_categories = []
    for sentence_labels in y:
        y_tmp = []
        for label in sentence_labels:
            category, polarity = label.split('_')
            y_tmp.append(category)
        y_categories.append(y_tmp)

    return y_categories


def create_fasttext_input_files(dataset_name, set_name, filename, X, y):
    with open(f"./data/{dataset_name}/{filename}.{set_name}", 'w') as write_file:
        for idx, sentence in enumerate(X):
            labels = ""
            for label in y[idx]:
                labels = labels + "__label__" + label + " "
            print(labels + sentence, file=write_file)


def create_fasttext_input_files_binary(dataset_name, set_name, filename, X, y):
    with open(f"./data/{dataset_name}/{filename}.{set_name}", 'w') as write_file:
        for idx, sentence in enumerate(X):
            label = "__label__" + y[idx] + " "
            print(label + sentence, file=write_file)


def create_as_input(X, y):
    new_X = []
    new_y = []
    for idx, sentence in enumerate(X):
        for label in y[idx]:
            cur_category, cur_polarity = label.split('_')
            new_sentence_text = sentence + "[sep]" + cur_category
            new_X.append(new_sentence_text)
            new_y.append([cur_polarity])

    return new_X, new_y


def create_as_input_x(X, y):
    new_X = []
    for idx, sentence in enumerate(X):
        for label in y[idx]:
            new_sentence_text = sentence + "[sep]" + label
            new_X.append(new_sentence_text)

    return new_X


def make_full_predictions(categories, polarities):
    return_lst = []

    idx_counter = 0
    for inner_category_list in categories:
        sentence_labels = []
        for category in inner_category_list:
            full_label = category + "_" + polarities[idx_counter][0]
            sentence_labels.append(full_label)
            idx_counter = idx_counter + 1
        return_lst.append(sentence_labels)

    return return_lst


def filter_categories(category, X, y):
    new_X = []
    new_y = []
    for idx, sentence in enumerate(X):
        for label in y[idx]:
            if category + "_positive" == label or category + "_negative" == label or category + "_neutral" == label:
                new_X.append(sentence)
                new_y.append(label.split('_')[1])

    return new_X, new_y


def onehot_to_labels_mapping(y_onehot, mapping_dict):
    new_y = []
    mapping_dict_reversed = {v: k for k, v in mapping_dict.items()}
    for outer_lst in y_onehot:
        tmp_y = []
        for idx, onehot_label in enumerate(outer_lst):
            if onehot_label == 1:
                tmp_y.append(mapping_dict_reversed[idx])
        new_y.append(tmp_y)

    return new_y


def probs_over_threshold(y_pred_probs, threshold):
    """
        Selects elements over given threshold from 2d y onehot array and converts it to onehot representation
    """
    y_new_onehot = []
    for sublist in y_pred_probs:
        y_tmp = []
        for prob_elem in sublist:
            if prob_elem > threshold:
                y_tmp.append(1)
            else:
                y_tmp.append(0)
        y_new_onehot.append(y_tmp)

    return y_new_onehot


def average_sentence_length(X):
    words_count = 0
    for sentence in X:
        words_count = words_count + len(sentence.split())

    return words_count / len(X)

