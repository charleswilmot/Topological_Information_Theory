import os
import numpy as np
from jax import random
from . import ndshape as nds


def cache(ndshape):
    print(ndshape)
    key = random.PRNGKey(0)
    n = 1000 ** ndshape._manifold_dimension
    if n * ndshape._embedding_dimension > 10 ** 8:
        raise ValueError("This is too much data")
    filepath = f'../cache/{ndshape.__class__.__name__}_md_{ndshape._manifold_dimension}_ed_{ndshape._embedding_dimension}.npy'
    if os.path.isfile(filepath):
        raise ValueError("This cache file already exists")
    np.save(filepath, ndshape.sample(key, n))


if __name__ == '__main__':
    ndshapes = (
        nds.MoebiusStrip(),
        nds.KleinTube(),
    )

    for ndshape in ndshapes:
        cache(ndshape)
