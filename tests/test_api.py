import fitfleet


def test_public_api():
    assert fitfleet.__version__
    for name in fitfleet.__all__:
        assert hasattr(fitfleet, name)
