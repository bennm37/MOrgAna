import pytest
# from PyQt5.QtWidgets import QApplication
from morgana.GUIs.mainwindow import morganaApp
from PyQt5.QtCore import Qt


# @pytest.fixture(scope="session")
# def app():
#     """Create a QApplication instance for the test session."""
#     return QApplication([])


@pytest.fixture
def main_window(qtbot):
    """Create the main window instance for each test."""
    main_window = morganaApp()
    qtbot.addWidget(main_window)
    return main_window


def test_initial_state(main_window):
    print("Testing Main Window")
    assert main_window.modelFolder == "-"
    assert main_window.imageFolder == "-"
    assert main_window.imageImportFolder == "-"
    assert main_window.maskFolder == "-"
    assert main_window.classifier is None
    assert main_window.scaler is None
    assert main_window.params == {
        "sigmas": [1, 2, 5, 15],
        "down_shape": 0.25,
        "edge_size": 2,
        "fraction": 0.5,
        "bias": 0.5,
        "feature_mode": "ilastik",
    }


def test_mask_tab_initial_state(main_window):
    assert main_window.isMask.isChecked() is False
    assert main_window.modelGroup.isVisible() is True
    assert main_window.importGroup.isVisible() is True


def test_change_mask_group(app, qtbot):
    qtbot.mouseClick(app.isMask, Qt.LeftButton)
    assert app.isMask.isChecked() is True
    # Add more assertions based on what changeMaskGroup should do


def test_select_model_folder(app, qtbot, mocker):
    mocker.patch(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        return_value="/path/to/model",
    )
    app.selectModelFolder()
    assert app.modelFolder == "/path/to/model"
    assert app.modelFolderSpace.text() == "/path/to/model"


def test_select_image_folder(app, qtbot, mocker):
    mocker.patch(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        return_value="/path/to/images",
    )
    app.selectImageFolder()
    assert app.imageFolder == "/path/to/images"
    assert app.imageFolderSpace.text() == "/path/to/images"


def test_train_model(app, qtbot, mocker):
    mocker.patch.object(app, "trainModel", return_value=None)
    qtbot.mouseClick(app.trainButton, Qt.LeftButton)
    app.trainModel.assert_called_once()


def test_predict(app, qtbot, mocker):
    mocker.patch.object(app, "predict", return_value=None)
    qtbot.mouseClick(app.predictButton, Qt.LeftButton)
    app.predict.assert_called_once()


def test_make_recap(app, qtbot, mocker):
    mocker.patch.object(app, "makeRecap", return_value=None)
    qtbot.mouseClick(app.recapButton, Qt.LeftButton)
    app.makeRecap.assert_called_once()


def test_open_inspection_window(app, qtbot, mocker):
    mocker.patch.object(app, "openInspectionWindow", return_value=None)
    qtbot.mouseClick(app.inspectButton, Qt.LeftButton)
    app.openInspectionWindow.assert_called_once()
