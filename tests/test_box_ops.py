import pytest
from groundingdino.util import box_ops
import torch
import numpy as np


@pytest.fixture(name="boxes")
def boxes_fixture():
    """
    Fixture that yields boxes in dino and numpy format and image shape
    """
    image_shape = (480, 640)
    num_boxes = 1000

    top_left_x = np.random.randint(low=1, high=639, size=(num_boxes, 1))
    top_left_y = np.random.randint(low=1, high=479, size=(num_boxes, 1))
    widths = np.random.randint(low=1, high=639, size=(num_boxes, 1))
    heights = np.random.randint(low=1, high=479, size=(num_boxes, 1))

    bottom_left_x = np.minimum(top_left_x + widths, image_shape[1] - 1)
    bottom_left_y = np.minimum(top_left_y + heights, image_shape[0] - 1)

    widths = bottom_left_x - top_left_x
    heights = bottom_left_y - top_left_y

    centre_x = top_left_x + widths / 2.0
    centre_y = top_left_y + heights / 2.0

    output = {}
    output["np"] = np.hstack((top_left_x, top_left_y, widths, heights))
    output["dino"] = torch.from_numpy(
        np.hstack(
            (
                centre_x / image_shape[1],
                centre_y / image_shape[0],
                widths / image_shape[1],
                heights / image_shape[0],
            )
        )
    )
    invalid = np.any(output["np"] <= 0, axis=1)
    if np.any(invalid):
        output["np"] = np.delete(output["np"], np.where(invalid), axis=0)
        output["dino"] = output["dino"][not torch.from_numpy(invalid)]

    output["shape"] = image_shape
    yield output


def test_dino2np(boxes):
    np_boxes = box_ops.dino2np(boxes["dino"], image_shape=boxes["shape"])

    assert np.allclose(np_boxes, boxes["np"]), "Incorrect numpy bounding boxes"


def test_dino2np_oob(boxes):
    """
    Tests for correct errors when box is out of image bounds
    """
    num_boxes = boxes["np"].shape[0]

    for i in range(num_boxes):
        dino_box = boxes["dino"][i]
        diff_x = boxes["shape"][0] - (dino_box[1] + dino_box[3] / 2)
        diff_y = boxes["shape"][1] - (dino_box[0] + dino_box[2] / 2)
        dino_box[3] += diff_x + 1
        dino_box[2] += diff_y + 1
        with pytest.raises(AssertionError, match="Bounding box out of dimension"):
            box_ops.dino2np(dino_box.unsqueeze(0), boxes["shape"])
