"""Describing usage of nelder_mead.py module"""
import numpy as np
from quadratic import function
import simplex


class NelderMead:
    """Class NelderMead"""
    best: np.array
    good: np.array
    worst: np.array
    cog: np.array
    alpha: float
    beta: float
    gamma: float
    max_iter: int
    simplex: np.array

    def __init__(self, alpha=1, beta=0.5, gamma=2, maxiter=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = maxiter

    def execute_nelder_mead(self, dim, *point):
        pbest, pgood, pworst = [], [], []
        if len(point) != 0:
            self.simplex = simplex.Simplex(dim, point)
        else:
            self.simplex = simplex.Simplex(dim)
        self.simplex.points.sort(key=function)
        self.best = self.simplex.points[0]
        self.worst = self.simplex.points[-1]
        self.good = self.simplex.points[-2]
        for i in range(self.max_iter):
            pbest.clear()
            pgood.clear()
            pworst.clear()
            self.cog = (1 / len(self.simplex.points)) \
                       * sum(self.simplex.points[:-1])
            xr = self.reflection(self.worst)
            if function(xr) < function(self.best):
                xe = self.expansion(xr)
                if function(xe) < function(xr):
                    self.worst = xe
                if function(xr) < function(xe):
                    self.worst = xr
            elif function(self.best) < function(xr) \
                                          < function(self.good):
                self.worst = xr
                xs = self.expansion(self.worst)
                if function(xs) < function(self.worst):
                    self.worst = xs
                elif function(xs) > function(self.worst):
                    self.shrink()
            elif function(self.good) < function(xr) \
                                     < function(self.worst):
                self.worst, xr = xr, self.worst
                xs = self.expansion(self.worst)
                if function(xs) < function(self.worst):
                    self.worst = xs
                elif function(xs) > function(self.worst):
                    self.shrink()
            elif function(self.worst) < function(xr):
                xs = self.expansion(self.worst)
                if function(xs) < function(self.worst):
                    self.worst = xs
                elif function(xs) > function(self.worst):
                    self.shrink()
            for j in range(dim):
                pbest.append(np.format_float_positional(self.best[j], min_digits=5, precision=5))
                pgood.append(np.format_float_positional(self.good[j], min_digits=5, precision=5))
                pworst.append(np.format_float_positional(self.worst[j], min_digits=5, precision=5))
            print(f"Iteration number {i + 1}: "
                  f"best - {pbest}, good - {pgood} "
                  f"and worst = {pworst}")
            if np.allclose(self.simplex.points[1:],
                           self.best, atol=0.1, rtol=0):
                return pbest, function(self.best)
        return pbest, function(self.best)

    def reflection(self, x):
        return (1 + self.alpha) * self.cog + self.alpha * x

    def expansion(self, x):
        return (1 - self.gamma) * self.cog + self.gamma * x

    def contraction(self, x):
        return self.beta * x + (1 - self.beta) * self.cog

    def shrink(self):
        for i in range(1, len(self.simplex.points)):
            self.simplex.points[i] \
              = self.best + (self.simplex.points[i] - self.best) / 2
        self.worst = self.simplex.points[-1]
        self.good = self.simplex.points[-2]
