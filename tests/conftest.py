import pytest


@pytest.fixture(scope="session")
def glance():
    """One tiny CPU model for the whole session (downloads only the Qwen3.5 config and processor)."""
    from glance_engine import Glance

    return Glance.tiny()


@pytest.fixture(scope="session")
def images():
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(0)
    return [Image.fromarray((rng.rand(h, w, 3) * 255).astype("uint8")) for w, h in ((320, 240), (200, 300), (256, 256))]
