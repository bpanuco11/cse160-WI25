import os
import random
from typing import List, Tuple
import numpy as np

matrix_sizes = [
    ((2, 2), (2, 2)),
    ((16, 16), (16, 16)),
    ((64, 64), (64, 64)),
    ((64, 128), (128, 64)),
    ((112, 48), (48, 16)),
    ((84, 84), (84, 84)),
    ((80, 99), (99, 128)), 
    ((67, 53), (53, 64)),
    ((29, 117), (117, 85)),
    ((191, 19), (19, 241))
]

def write_to_file(filename : str, M : np.ndarray):
    assert(M.ndim == 2)
    with open(filename, "w") as f:
        rows, cols = M.shape
        f.write(f"# ({rows}, {cols})\n")
        for row in M:
            f.write(" ".join(map(str, row)) + "\n")

for i, size in enumerate(matrix_sizes):
    A = np.random.randint(-100, 100, size[0]).T
    B = np.random.randint(-100, 100, size[1])

    C = A.T @ B

    write_to_file(f"{i}/input0.raw", A)
    write_to_file(f"{i}/input1.raw", B)
    write_to_file(f"{i}/output.raw", C)

