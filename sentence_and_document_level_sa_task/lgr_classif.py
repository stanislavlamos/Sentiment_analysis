from sklearn.linear_model import LogisticRegression
import tensorflow_hub as hub
import utils


use_dan_path = "../pretrained_vectors/use_dan"
use_transformers_path = "../pretrained_vectors/use_transformer"


def lgr_binary_classif(X_train, y_train, X_test, encoder_type):
    use_encoder = None
    if encoder_type == "dan":
        use_encoder = hub.load(use_dan_path)
    elif encoder_type == "trans":
        use_encoder = hub.load(use_transformers_path)

    X_test_embed = []
    for idx, test_sample in enumerate(X_test):
        print(f"{idx+1}/{len(X_test)}")
        X_test_embed.append(use_encoder([test_sample]).numpy().tolist()[0])

    X_train_embed = []
    for idx, train_sample in enumerate(X_train):
        print(f"{idx + 1}/{len(X_train)}")
        X_train_embed.append(use_encoder([train_sample]).numpy().tolist()[0])

    lgr_classif = LogisticRegression(random_state=0).fit(X_train_embed, y_train)
    lgr_classes = lgr_classif.classes_
    y_pred = [lgr_classes[utils.argmax(pred_probs.tolist())] for pred_probs in lgr_classif.predict_proba(X_test_embed)]

    return y_pred
