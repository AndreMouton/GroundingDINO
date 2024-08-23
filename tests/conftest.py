import os
from pathlib import Path
import pytest
from groundingdino.perceiver.gdino import GroundingDinoPerceiver


@pytest.fixture(scope="session", autouse=True, name="model")
def model_setup_fixture():
    """Fixture to setup gdino perceiver model"""

    cfg_file = Path(
        f"{os.path.abspath(os.path.dirname(__file__))}/test_data/config/test_config.yaml"
    )
    gdino_model = GroundingDinoPerceiver(cfg_file)
    yield gdino_model
