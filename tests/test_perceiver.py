import os
import pytest
import cv2
import torch
import numpy as np

DATA_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/test_data/"
TEST_IMAGE = f"{DATA_DIR}/cat_dog.jpeg"
CFG = f"{DATA_DIR}/config/test_config.yaml"


@pytest.fixture(name="bboxes")
def bbox_fixture():
    """Fixture to init ground truth bounding boxes for each class"""

    bboxes = {
        "cat": np.array([5.8697510e-01, 3.1947714e02, 5.6215698e02, 5.5766144e02]),
        "dog": np.array([3.9465491e02, 1.1667786e01, 7.4542810e02, 8.8775739e02]),
    }
    yield bboxes


@pytest.fixture(name="input_data")
def input_fixture():
    """Fixture to construct input dict"""
    input_dict = {}
    image = cv2.imread(TEST_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_dict["IMAGE"] = image

    input_dict["PROMPT"] = ["cat", "dog"]

    yield input_dict


def test_no_image(model):
    """Tests for assertion error when no image provided"""
    no_image = {"PROMPT": ["cat", "dog"]}

    with pytest.raises(
        AssertionError, match="GroundingDINO error: input dict must contain IMAGE"
    ):
        model(no_image)


def test_no_prompt(model, input_data):
    """Tests for assertion error when no prompt provided"""
    no_prompt = {"IMAGE": input_data["IMAGE"]}

    with pytest.raises(
        AssertionError, match="GroundingDINO error: input dict must contain PROMPT"
    ):
        model(no_prompt)


def test_invalid_image(model, input_data):
    """Tests for assertion error when invalid image provided"""
    invalid_image = {
        "PROMPT": input_data["PROMPT"],
        "IMAGE": torch.from_numpy(input_data["IMAGE"]),
    }

    with pytest.raises(
        AssertionError, match="GroundingDINO error: input image must be numpy array"
    ):
        model(invalid_image)


def test_invalid_prompt(model, input_data):
    """Tests for assertion error when invalid prompt provided"""
    invalid_prompt = {
        "PROMPT": input_data["IMAGE"],
        "IMAGE": input_data["IMAGE"],
    }

    with pytest.raises(
        AssertionError, match="GroundingDINO error: prompt must be list of strings"
    ):
        model(invalid_prompt)


def test_valid_input(model, input_data, bboxes):
    """Tests for correct detection with valid input"""
    result = model(input_data)

    # Assert on classes
    classes = ["cat", "dog"]
    assert sorted(result.classes) == classes, "Invalid classes"
    assert not result.indices, "Indices should be none for detection only"  # No indices

    for i in range(result.bboxes.shape[0]):
        assert np.allclose(
            result.bboxes[i], bboxes[result.classes[i]]
        ), f"Invalid bounding box for class {result.classes[i]}"
