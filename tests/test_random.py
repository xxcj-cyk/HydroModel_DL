import random
import numpy as np
import torch
from trainers.trainer import set_random_seed


def test_set_random_seed():
    seed = 42

    # Set random seed
    set_random_seed(seed)

    # Generate random numbers
    random1 = random.random()
    np_random1 = np.random.rand()
    torch_random1 = torch.rand(1).item()

    # Set the same seed again
    set_random_seed(seed)

    # Generate random numbers again
    random2 = random.random()
    np_random2 = np.random.rand()
    torch_random2 = torch.rand(1).item()

    # Assert that the results are the same
    assert random1 == random2, f"Random results differ: {random1} vs {random2}"
    assert np.isclose(
        np_random1, np_random2
    ), f"NumPy results differ: {np_random1} vs {np_random2}"
    assert torch.isclose(
        torch.tensor(torch_random1), torch.tensor(torch_random2)
    ), f"Torch results differ: {torch_random1} vs {torch_random2}"

    print("All tests passed. Random seed setting is consistent.")


if __name__ == "__main__":
    test_set_random_seed()
