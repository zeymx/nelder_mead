import numpy as np


class Simplex:
    points: list

    def __init__(self, dim, *args):
        defaultargs = np.random.randint(1, 10, dim)
        args = args + (None,) * len(defaultargs)
        defaultargs = np.array(list(map(lambda x, y: y if y is not None else x, defaultargs, args)))
        unit_vectors = np.eye(dim)
        self.points = list()
        self.points.append(defaultargs)
        for i in range(1, dim):
            coef = 10
            curr_point = defaultargs + coef * unit_vectors[i]
            self.points.append(curr_point)
