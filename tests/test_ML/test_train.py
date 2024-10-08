from morgana.cli.train import train_model
from tests.data_utils import create_test_model_folder


def test_train_logistic():
    model = "logistic"
    model_folder = create_test_model_folder(model, from_pretrained=False)
    train_model(model, model_folder=model_folder, epochs=2, steps_per_epoch=2)


def test_train_mlp():
    model = "MLP"
    model_folder = create_test_model_folder(model, from_pretrained=False)
    train_model(model, model_folder=model_folder, epochs=2, steps_per_epoch=2)


def test_train_unet():
    model = "unet"
    model_folder = create_test_model_folder(model, from_pretrained=False)
    train_model(model, model_folder=model_folder, epochs=2, steps_per_epoch=2)


if __name__ == "__main__":
    test_train_logistic()
    test_train_mlp()
    test_train_unet()
