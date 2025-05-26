import tempfile
import os
import shutil
from morgana.MLModel.io import new_model, save_model
from morgana.CLI.train import train_model


def create_test_model_folder(model, from_pretrained=False):
    assert model in ["unet", "logistic", "MLP"]
    root = tempfile.mkdtemp()
    model_folder = os.path.join(root, "model")
    model_tiny = "./tests/test_data/model_tiny"
    if from_pretrained:
        shutil.copytree(model_tiny, model_folder)
        pretrained = f"./tests/test_data/pretrained/{model}"
        if not check_pretrained(pretrained, model):
            print(f"Pretrained model for {model} not found. Creating new model.")
            create_pretrained(model)
        [shutil.copy(os.path.join(pretrained, f), model_folder) for f in os.listdir(pretrained)]
    else:
        new_model(model_folder, model)
        [
            shutil.copy(os.path.join(model_tiny, "trainingset", f), os.path.join(model_folder, "trainingset"))
            for f in os.listdir(os.path.join(model_tiny, "trainingset"))
        ]
        [
            shutil.copy(os.path.join(model_tiny, "testset", f), os.path.join(model_folder, "testset"))
            for f in os.listdir(os.path.join(model_tiny, "testset"))
        ]
    return model_folder


def check_pretrained(path, model):
    valid = True
    if model == "logistic":
        if not os.path.exists(f"{path}/classifier.pkl"):
            valid = False
    else:
        if not os.path.exists(f"{path}/classifier.keras"):
            valid = False
    if not os.path.exists(f"{path}/scaler.pkl"):
        valid = False
    if not os.path.exists(f"{path}/params.json"):
        valid = False
    return valid


def create_pretrained(model):
    model_folder = create_test_model_folder(model, from_pretrained=False)
    classifier, scaler, params = train_model(model, model_folder=model_folder, epochs=10, steps_per_epoch=2)
    save_model(f"./tests/test_data/pretrained/{model}", classifier, scaler, **params)


if __name__ == "__main__":
    for model in ["logistic", "unet", "MLP"]:
        create_pretrained()
