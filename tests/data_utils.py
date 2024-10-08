import tempfile
import os
import shutil
from morgana.MLModel.io import new_model


def create_test_model_folder(model, from_pretrained=False):
    assert model in ["unet", "logistic", "MLP"]
    root = tempfile.mkdtemp()
    model_folder = os.path.join(root, "model")
    model_tiny = "./tests/test_data/model_tiny"
    if from_pretrained:
        os.mkdir(model_folder)
        os.copytree(model_tiny, model_folder)
        pretrained = f"./tests/test_data/pretrained/{model}"
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
