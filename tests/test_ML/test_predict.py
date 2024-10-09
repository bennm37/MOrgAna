import pytest
from morgana.cli.predict import predict_single_image, predict_batch, predict_folder
from morgana.DatasetTools.io import get_image_list
from morgana.MLModel.io import load_model
from tests.data_utils import create_test_model_folder
import os


@pytest.mark.parametrize("model", ["logistic", "MLP", "unet"])
def test_predict(model):
    model_folder = create_test_model_folder(model, from_pretrained=True)
    image_folder = f"{model_folder}/data"
    classifier, scaler, params = load_model(model_folder, model=model)
    flist = get_image_list(image_folder, string_filter="_GT", mode_filter="exclude")
    predict_folder(image_folder, classifier, scaler, params, model=model)
    assert os.path.exists(f"{image_folder}/result_segmentation")
    flist_rs = get_image_list(f"{image_folder}/result_segmentation", string_filter="_classifier", mode_filter="include")
    assert len(flist) == len(flist_rs)
    for f_rs, f in zip(flist_rs, flist):
        assert f_rs.split("/")[-1].split("_")[0] == f.split("/")[-1].split(".")[0]
    predict_folder(image_folder, classifier, scaler, params)
    assert len(flist) == len(flist_rs)
    for f_rs, f in zip(flist_rs, flist):
        assert f_rs.split("/")[-1].split("_")[0] == f.split("/")[-1].split(".")[0]


if __name__ == "__main__":
    test_predict("logistic")
    test_predict("MLP")
    test_predict("unet")
