import numpy as np
from markowitz_no_short import simulated_annealing

mean = np.array([0.5, 0.1, 0.4])
cov = np.array([[1, 0.5, -0.3], [0.5, 1, -0.1], [-0.3, -0.1, 1]])

simulated_annealing(mean, cov, q=5, temperature=5, its=100000)
