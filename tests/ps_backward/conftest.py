import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
