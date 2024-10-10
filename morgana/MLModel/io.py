import joblib
import os
import json
import morgana.DatasetTools.io as ioDT
import random


def new_model(model_folder, model="logistic"):
    try:
        to_unicode = unicode  # type: ignore
    except NameError:
        to_unicode = str
    model_folder = os.path.abspath(model_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    else:
        raise ValueError("Model folder already exists.")
    os.mkdir(os.path.join(model_folder, "trainingset"))
    if model not in ["logistic", "MLP", "unet"]:
        raise ValueError("Model not recognized.")
    # current file path
    current_path = os.path.dirname(os.path.realpath(__file__))
    default_path = os.path.join(current_path, f"default_params/{model}.json")
    default_params = json.load(open(default_path))
    with open(os.path.join(model_folder, "params.json"), "w") as f:
        str_ = json.dumps(
            default_params,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        f.write(to_unicode(str_))
    print(f"Created new {model} model at {model_folder}")
    print("Populate the trainingset folder with images and edit the params.json file.")
    return model_folder


def split_test(model_folder, fraction=0.1):
    assert folder_labeled(f"{model_folder}/trainingset")
    assert not os.path.exists(os.path.join(model_folder, "testset")), f"{model_folder}/testset already exists."
    os.mkdir(os.path.join(model_folder, "testset"))
    flist_GT = ioDT.get_image_list(
        os.path.join(model_folder, "trainingset"), string_filter="_GT", mode_filter="include"
    )
    flist_in = ioDT.get_image_list(
        os.path.join(model_folder, "trainingset"), string_filter="_GT", mode_filter="exclude"
    )
    n_img = len(flist_GT)
    n_test = int(n_img * fraction)
    for i in range(n_test):
        f = random.choice(flist_in)
        f_gt = f.replace(".tif", "_GT.tif")
        os.rename(f, os.path.join(model_folder, "testset", os.path.basename(f)))
        os.rename(f_gt, os.path.join(model_folder, "testset", os.path.basename(f_gt)))
    print(f"Split {n_test} images to {model_folder}/testset")
    return True


def folder_labeled(folder):
    assert os.path.exists(folder), f"{folder} does not exist."
    assert len(os.listdir(folder)) > 0, f"{folder} is empty."
    flist_GT = ioDT.get_image_list(
        folder, string_filter="_GT", mode_filter="include"
    )
    flist_in = ioDT.get_image_list(
        folder, string_filter="_GT", mode_filter="exclude"
    )
    assert len(flist_GT) == len(flist_in), "Number of GT and input images do not match."
    for f in flist_in:
        fn, ext = os.path.splitext(f)
        mask_name = fn + "_GT" + ext
        assert os.path.exists(mask_name), f"Mask {mask_name} not found."
    print(f"All images in {folder} are labeled.")
    return True


def save_model(
    model_folder,
    classifier,
    scaler,
    **params,
):
    """
    save a previously generated machine learning model in the "model_folder" input path:
    * model_folder/classifier.pkl: logistic classifier model
    * model_folder/scaler.pkl: scaler used to normalize the trainingset
    * model_folder/params.json: parameters used for training

    """

    # Make it work for Python 2+3 and with Unicode
    try:
        to_unicode = unicode  # type: ignore
    except NameError:
        to_unicode = str
    model = params.get("model", "logistic")
    if model == "logistic":
        joblib.dump(classifier, os.path.join(model_folder, "classifier.pkl"))
    else:
        classifier.save(os.path.join(model_folder, "classifier.keras"))

    joblib.dump(scaler, os.path.join(model_folder, "scaler.pkl"))

    params["model"] = model
    with open(os.path.join(model_folder, "params.json"), "w", encoding="utf8") as f:
        str_ = json.dumps(
            params,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        f.write(to_unicode(str_))


def load_model(model_folder):
    """
    load a previously saved machine learning model from the "model_folder" input path:
    * model_folder/classifier.pkl: logistic classifier model
    * model_folder/scaler.pkl: scaler used to normalize the trainingset
    * model_folder/params.json: parameters used for training

    """
    scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
    params = load_params(model_folder)
    model = params.get("model", "logistic")
    if model == "logistic":
        try:
            classifier = joblib.load(os.path.join(model_folder, "classifier.pkl"))
        except:  # noqa E722 # TODO find the correct exception
            return None, None, None
    else:
        from tensorflow import keras

        try:
            classifier = keras.models.load_model(os.path.join(model_folder, "classifier.keras"))
        except:  # noqa E722 # TODO find the correct exception
            return None, None, None
    return classifier, scaler, params


def load_params(model_folder):
    with open(os.path.join(model_folder, "params.json"), "r") as f:
        params = json.load(f)
    return params
