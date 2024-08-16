from pathlib import Path
from typing import Any

import cv2
from perceiver.defs import PerceiverOutput
from perceiver.perceiver_factory import PerceiverFactory
from perceiver.perceiver_base import PerceiverBase

from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate


@PerceiverFactory.register("gdino")
class GroundingDinoPerceiver(PerceiverBase):
    def __init__(self, config_file: Path):
        """
        Inits grounding dino model from config file

        Args:
            config_file (Path): Full path to yaml config file
        """

        super().__init__(config_file)

        # Load grounding dino model
        self.model = load_model(
            model_config_path=self.config.model.config_path,
            model_checkpoint_path=self.config.model.ckpt,
            device="cuda",
        )

        self._image_transforms = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, input_dict: dict[Any, Any]) -> PerceiverOutput:
        """
        Calls core gdino on input data and prompt
        Args:
            input_dict (dict[Any, Any]): Dict with "IMAGE" and "PROMPTS" keys.

        Returns (PerceiverOutput): Resulting bounding box detections

        """
        # TODO add checks
        input_image, _ = self._image_transforms(
            Image.fromarray(input_dict["IMAGE"]), None
        )
        input_image = input_image.to("cuda")
        prompt = " . ".join(input_dict["PROMPT"])
        prompt = f"{prompt}."

        boxes, logits, phrases = predict(
            model=self.model,
            image=input_image,
            caption=prompt,
            box_threshold=self.config.model.box_thresh,
            text_threshold=self.config.model.text_thresh,
        )

        # TODO populate output

        annotated_frame = annotate(
            image_source=input_dict["IMAGE"],
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey()


if __name__ == "__main__":
    image = cv2.imread("/home/arahman/repos/GroundingDINO/.asset/cat_dog.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prompt = ["cat", "dog"]

    gdino = GroundingDinoPerceiver(
        config_file=Path("/home/arahman/repos/perceiver/demos/configs/gdino.yaml")
    )
    input_dict = {}
    input_dict["IMAGE"] = image
    input_dict["PROMPT"] = prompt
    gdino(input_dict)