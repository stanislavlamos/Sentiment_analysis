import sklearn
import tensorflow_hub as hub
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import utils


use_dan_path = "../pretrained_vectors/use_dan"
use_transformers_path = "../pretrained_vectors/use_transformer"


def logistic_regression_categories(X_test, y_test, X_train, y_train, mapping_dict_categories, encoder_type, threshold):
    use_encoder = None
    if encoder_type == "dan":
        use_encoder = hub.load(use_dan_path)
    elif encoder_type == "trans":
        use_encoder = hub.load(use_transformers_path)

    X_test_embed = use_encoder(X_test)
    X_train_embed = use_encoder(X_train)
    n_classes = len(mapping_dict_categories.keys())
    y_train_onehot = utils.convert_to_onehot(utils.delete_polarities(y_train), mapping_dict_categories, n_classes)
    y_test_onehot = utils.convert_to_onehot(utils.delete_polarities(y_test), mapping_dict_categories, n_classes)

    lgr = MultiOutputClassifier(estimator=LogisticRegression(random_state=0, max_iter=1000)).fit(
        X_train_embed,
        y_train_onehot)
    y_preds_probs = lgr.predict_proba(X_test_embed)

    y_preds_onehot = []
    threshold = threshold
    for i in range(len(y_test)):
        sentence_pred_probs = []

        for j in range(n_classes):
            sentence_pred_probs.append(y_preds_probs[j][i][1])
        sentence_pred_onehot = [1 if el > threshold else 0 for el in sentence_pred_probs]
        y_preds_onehot.append(sentence_pred_onehot)

    # threshold = 0
    # rpr = []
    # is_vals = []
    # for i in range(100):
    #     y_preds_onehot = []
    #     threshold = threshold + 0.01
    #     for i in range(len(y_test)):
    #         sentence_pred_probs = []
    #         for j in range(n_classes):
    #             sentence_pred_probs.append(y_preds_probs[j][i][1])
    #         sentence_pred_onehot = [1 if el > threshold else 0 for el in sentence_pred_probs]
    #         y_preds_onehot.append(sentence_pred_onehot)
    #     rpr_dict = sklearn.metrics.classification_report(y_test_onehot, y_preds_onehot, zero_division=0, target_names=mapping_dict_categories.keys(), output_dict=True)
    #     rpr.append(rpr_dict["micro avg"]["f1-score"])
    #     is_vals.append(threshold)
    # print(is_vals[utils.argmax(rpr)])
    # print(max(rpr))

    return y_preds_onehot


def logistic_regression_polarities(X_test, y_preds_categories, y_test_true, X_train, y_train, mapping_dict_categories,
                                   mapping_dict_polarities, encoder_type):
    use_encoder = None
    if encoder_type == "dan":
        use_encoder = hub.load(use_dan_path)
    else:  # type == trans
        use_encoder = hub.load(use_transformers_path)

    X_test_embed = use_encoder(X_test)
    category_logistic_regression = []
    for category in mapping_dict_categories.keys():
        X_train_polarities, y_train_polarities = utils.filter_categories(category, X_train, y_train)
        y_train_polarities = [mapping_dict_polarities[el] for el in y_train_polarities]
        X_train_polarities_embed = use_encoder(X_train_polarities)
        lgr = LogisticRegression(random_state=0).fit(X_train_polarities_embed, y_train_polarities)
        category_logistic_regression.append(lgr)

    mapping_dict_polarities_reversed = {v: k for k, v in mapping_dict_polarities.items()}
    y_full_predicted = []
    for idx, sentence in enumerate(X_test):
        y_tmp = []
        for cur_category in y_preds_categories[idx]:
            category_lgr = category_logistic_regression[mapping_dict_categories[cur_category]]
            predicted_polarity = mapping_dict_polarities_reversed[category_lgr.predict(use_encoder([sentence]))[0]]
            y_tmp.append(cur_category + "_" + predicted_polarity)
        y_full_predicted.append(y_tmp)

    return y_full_predicted
