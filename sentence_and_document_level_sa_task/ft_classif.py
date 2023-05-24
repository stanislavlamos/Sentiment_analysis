import fasttext


pretrained_vectors_path = "../pretrained_vectors/cc.en.300.vec"


def ft_classification_multilabel(dataset_name, x_test, lr=0.05, epoch=10, dim=300, wordNgrams=1, autotune=False, twostep=""):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    if autotune:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}" + twostep + ".train",
                                          loss='ova',
                                          dim=300,
                                          pretrained_vectors=pretrained_vectors_path,
                                          autotuneValidationFile=f"./data/{dataset_name}/{dataset_name}" + twostep + ".valid")

        print("Generated hyperparameters")
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

    else:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}" + twostep + ".train",
                                          loss='ova', lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams)

        print("Hyperparameters")
        print(f"Lr: {lr}")
        print(f"Epochs: {epoch}")
        print(f"Dim: {dim}")
        print("Loss: ova")

    samples_number_val, precision_val, recall_val = model.test(
        f"./data/{dataset_name}/{dataset_name}" + twostep + ".valid")
    print(f"Number of validation samples: {samples_number_val}")
    print(f"Precision on validation set: {precision_val}")
    print(f"Recall on validation set: {recall_val}")

    y_pred = model.predict(x_test, k=-1, threshold=0.18)
    y_pred_probs = y_pred[1]

    y_pred_shortened = []
    for possible_labels in y_pred[0]:
        y_pred_shortened_inner = []
        for cur_label in possible_labels:
            cur_label_cleared = cur_label.replace('__label__', '')
            y_pred_shortened_inner.append(cur_label_cleared)
        y_pred_shortened.append(y_pred_shortened_inner)

    return y_pred_shortened, y_pred_probs


def ft_classification_onelabel(dataset_name, x_test, lr=0.05, epoch=10, dim=300, wordNgrams=1, autotune=False, twostep=""):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    if autotune:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}" + twostep + ".train",
                                          pretrained_vectors=pretrained_vectors_path,
                                          dim=300,
                                          autotuneValidationFile=f"./data/{dataset_name}/{dataset_name}" + twostep + ".valid")

        print("Generated hyperparameters")
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

    else:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{dataset_name}" + twostep + ".train",
                                          loss='hs', lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams)#, pretrained_vectors=pretrained_vectors_path)

        print("Hyperparameters")
        print(f"Lr: {lr}")
        print(f"Epochs: {epoch}")
        print(f"Dim: {dim}")
        print("Loss: ova")

    samples_number_val, precision_val, recall_val = model.test(
        f"./data/{dataset_name}/{dataset_name}" + twostep + ".valid")
    print(f"Number of validation samples: {samples_number_val}")
    print(f"Precision on validation set: {precision_val}")
    print(f"Recall on validation set: {recall_val}")

    y_pred = model.predict(x_test)
    y_pred_probs = y_pred[1]

    y_pred_shortened = []
    for possible_labels in y_pred[0]:
        y_pred_shortened_inner = []
        for cur_label in possible_labels:
            cur_label_cleared = cur_label.replace('__label__', '')
            y_pred_shortened_inner.append(cur_label_cleared)
        y_pred_shortened.append(y_pred_shortened_inner)

    return y_pred_shortened, y_pred_probs
