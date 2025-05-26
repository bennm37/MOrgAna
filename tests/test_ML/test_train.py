import pytest
from morgana.CLI.train import train_model
from tests.data_utils import create_test_model_folder

MODELS = ["logistic", "MLP", "unet"]


@pytest.mark.parametrize("model", MODELS)
def test_train(model):
    model_folder = create_test_model_folder(model, from_pretrained=False)
    train_model(model, model_folder=model_folder, epochs=2, steps_per_epoch=2)


if __name__ == "__main__":
    for model in MODELS:
        test_train(model)
