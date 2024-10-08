import joblib
import os
import json


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


def save_model(
    model_folder,
    classifier,
    scaler,
    model="logistic",
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


def load_model(model_folder, model="logistic"):
    """
    load a previously saved machine learning model from the "model_folder" input path:
    * model_folder/classifier.pkl: logistic classifier model
    * model_folder/scaler.pkl: scaler used to normalize the trainingset
    * model_folder/params.json: parameters used for training

    """
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

    scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
    with open(os.path.join(model_folder, "params.json"), "r") as f:
        params = json.load(f)

    # patch to take into account the old definition of down_shape
    if params["down_shape"] == 500:
        params["down_shape"] = 500.0 / 2160.0
    return classifier, scaler, params
