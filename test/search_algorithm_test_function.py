import numpy as np
# test function
class TestFunction:

    def optimal_value(self):
        raise NotImplementedError

    def objective_function(self, x):
        raise NotImplementedError


class SumCategoryTestFunction(TestFunction):
    def __init__(self, category):
        self.category = category

    def objective_function(self, sample):
        return np.sum(sample)

    def epoch_objective_function(self, sample, epoch):
        return np.sum(sample) * np.log10(epoch)

    def optimal_value(self):
        return np.array(self.category) - 1

    def l2_distance(self, sample):
        return np.sum((self.optimal_value() - sample) ** 2)