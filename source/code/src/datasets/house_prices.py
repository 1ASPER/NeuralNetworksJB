"""Synthetic house prices: features on very different scales."""
import torch

FEATURE_NAMES = ['area, m2', 'rooms', 'year built', 'distance to centre, km', 'floor']


def make_house_prices(n_houses=6000, seed=0):
    """Random houses and their prices.

    Returns ``(features, prices)``: ``float [n, 5]`` in the order of
    ``FEATURE_NAMES`` (areas 30..300, years 1900..2020, ...) and
    ``float [n]``, prices of about 1e5..1e6.
    """
    generator = torch.Generator().manual_seed(seed)
    area = 30 + 270 * torch.rand(n_houses, generator=generator)
    rooms = torch.randint(1, 9, (n_houses,), generator=generator).float()
    year = 1900 + 120 * torch.rand(n_houses, generator=generator)
    distance = 0.5 + 49.5 * torch.rand(n_houses, generator=generator)
    floor = torch.randint(0, 31, (n_houses,), generator=generator).float()
    noise = torch.randn(n_houses, generator=generator)
    prices = (3000 * area * torch.exp(-distance / 25) + 15000 * rooms
              + 800 * (year - 1900) + 2000 * floor.sqrt() + 20000 * noise)
    return torch.stack([area, rooms, year, distance, floor], dim=1), prices


class Standardizer:
    """Shift and scale values to mean 0 and standard deviation 1.

    ``fit(values)`` remembers ``mean`` and ``std`` over the first dimension
    (one per column); ``transform(values) = (values - mean) / std``;
    ``inverse_transform`` undoes it.
    """

    def __init__(self):
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(1.0)

    def fit(self, values):
        ## YOUR CODE HERE
        return self

    def transform(self, values):
        ## YOUR CODE HERE
        return values

    def inverse_transform(self, values):
        ## YOUR CODE HERE
        return values


class HousePriceDataset(torch.utils.data.Dataset):
    """Items ``{'features': float [5], 'price': float}``."""

    def __init__(self, features, prices):
        self.features = features
        self.prices = prices

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, index):
        return {'features': self.features[index], 'price': self.prices[index]}
