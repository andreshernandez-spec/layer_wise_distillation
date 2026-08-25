import pytest


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", help="structural tests only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip = pytest.mark.skip(reason="--fast")
        for it in items:
            if "slow" in it.keywords:
                it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: statistical or model-loading tests")
