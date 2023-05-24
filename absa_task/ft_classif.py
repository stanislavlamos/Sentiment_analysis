import fasttext


pretrained_vectors_path = "../pretrained_vectors/cc.en.300.vec"


def ft_classification_multilabel(dataset_name, x_test, train_file_name, valid_file_name, threshold, lr=0.05, epoch=10, dim=300, wordNgrams=1, autotune=False):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    if autotune:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{train_file_name}",
                                          loss="ova",
                                          dim=300,
                                          pretrainedVectors=pretrained_vectors_path,
                                          # autotuneModelSize="100G",
                                          autotuneValidationFile=f"./data/{dataset_name}/{valid_file_name}")

        print("Generated hyperparameters")
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

    else:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{train_file_name}",
                                          loss='ova', lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams, pretrainedVectors=pretrained_vectors_path)

        print("Hyperparameters")
        print(f"Lr: {lr}")
        print(f"Epochs: {epoch}")
        print(f"Dim: {dim}")
        print("Loss: ova")

    samples_number_val, precision_val, recall_val = model.test(f"./data/{dataset_name}/{valid_file_name}")
    print(f"Number of validation samples: {samples_number_val}")
    print(f"Precision on validation set: {precision_val}")
    print(f"Recall on validation set: {recall_val}")

    y_pred = model.predict(x_test, k=-1, threshold=threshold)
    y_pred_probs = y_pred[1]

    y_pred_shortened = []
    for possible_labels in y_pred[0]:
        y_pred_shortened_inner = []
        for cur_label in possible_labels:
            cur_label_cleared = cur_label.replace('__label__', '')
            y_pred_shortened_inner.append(cur_label_cleared)
        y_pred_shortened.append(y_pred_shortened_inner)

    return y_pred_shortened, y_pred_probs


def ft_classification_onelabel(dataset_name, x_test, train_file_name, valid_file_name, lr=0.05, epoch=10, dim=300, wordNgrams=1, autotune=False):
    """
        Function to train fasttext supervised classification model and return predicted labels from test set
    """
    print(f"Starting fasttext model training on {dataset_name} dataset")

    if autotune:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{train_file_name}",
                                          pretrained_vectors=pretrained_vectors_path,
                                          dim=300,
                                          # autotuneModelSize="100G",
                                          autotuneValidationFile=f"./data/{dataset_name}/{valid_file_name}")

        print("Generated hyperparameters")
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

    else:
        model = fasttext.train_supervised(input=f"./data/{dataset_name}/{train_file_name}",
                                          loss='hs', lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams, pretrainedVectors=pretrained_vectors_path)

        print("Hyperparameters")
        print(f"Lr: {lr}")
        print(f"Epochs: {epoch}")
        print(f"Dim: {dim}")
        print("Loss: ova")

    samples_number_val, precision_val, recall_val = model.test(f"./data/{dataset_name}/{valid_file_name}")
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
