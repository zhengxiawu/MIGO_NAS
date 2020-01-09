import numpy as np
import tqdm
from search_algorithm import Category_DDPNAS, Category_MDENAS, Category_SNG, \
    Category_ASNG, Category_Dynamic_ASNG, Category_Dynamic_SNG, Category_Dynamic_SNG_V3
from test.search_algorithm_test_function import SumCategoryTestFunction


def get_optimizer(name, category):
    if name == 'DDPNAS':
        return Category_DDPNAS.CategoricalDDPNAS(category, 10)
    elif name == 'MDENAS':
        return Category_MDENAS.CategoricalMDENAS(category, 0.01)
    elif name == 'SNG':
        return Category_SNG.SNG(categories=category)
    elif name == 'ASNG':
        return Category_ASNG.ASNG(categories=category)
    elif name == 'dynamic_ASNG':
        return Category_Dynamic_ASNG.Dynamic_ASNG(categories=category, step=10, pruning=True)
    elif name == 'dynamic_SNG':
        return Category_Dynamic_SNG.Dynamic_SNG(categories=category, step=10,
                                                pruning=False, sample_with_prob=False)
    elif name == 'dynamic_SNG_V3':
        return Category_Dynamic_SNG_V3.Dynamic_SNG(categories=category, step=10,
                                                   pruning=True, sample_with_prob=False,
                                                   utility_function='log', utility_function_hyper=0.4,
                                                   momentum=True, gamma=0.9)
    else:
        raise NotImplementedError


category = [10]*10
test_function = SumCategoryTestFunction(category)
optimizer_name = 'dynamic_SNG_V3'

# distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
distribution_optimizer = get_optimizer(optimizer_name, category)
runing_times = 1000
runing_epochs = 1000
record = {
    'objective': np.zeros([runing_times, runing_epochs]) - 1,
    'l2_distance': np.zeros([runing_times, runing_epochs]) - 1,
}
for i in tqdm.tqdm(range(runing_times)):
    for j in range(runing_epochs):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        sample = distribution_optimizer.sampling_index()
        objective = test_function.objective_function(sample)
        distribution_optimizer.record_information(sample, objective)
        distribution_optimizer.update()
        current_best = np.argmax(distribution_optimizer.p_model.theta, axis=1)
        distance = test_function.l2_distance(current_best)
        record['objective'][i, j] = objective
        record['l2_distance'][i, j] = distance
    distribution_optimizer = get_optimizer(optimizer_name, category)
mean_obj = np.mean(record['objective'], axis=0)
mean_distance = np.mean(record['l2_distance'], axis=0)
print(mean_distance)
pass
