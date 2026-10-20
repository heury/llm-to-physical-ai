import pytest
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor


@pytest.fixture(scope="session")
def siglip_dir(tmp_path_factory):
    """Small actual HF SigLIP, random weights only for fast offline tests."""
    path = tmp_path_factory.mktemp("siglip")
    model = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=64,
            patch_size=16,
        )
    )
    model.save_pretrained(path)
    SiglipImageProcessor(size={"height": 64, "width": 64}).save_pretrained(path)
    return path
