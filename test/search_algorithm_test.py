import numpy as np
import tqdm
import pdb
import time
import os
from search_algorithm import Category_DDPNAS, Category_MDENAS, Category_SNG, \
    Category_ASNG, Category_Dynamic_ASNG, Category_Dynamic_SNG, Category_Dynamic_SNG_V3, \
    Category_DDPNAS_V3, Category_DDPNAS_V2
from test.search_algorithm_test_function import SumCategoryTestFunction, EpochSumCategoryTestFunction


def get_optimizer(name, category):
    if name == 'DDPNAS':
        return Category_DDPNAS.CategoricalDDPNAS(category, 3)
    elif name == 'DDPNAS_V2':
        return Category_DDPNAS_V2.CategoricalDDPNASV2(category, 3)
    elif name == 'DDPNAS_V3':
        return Category_DDPNAS_V3.CategoricalDDPNASV3(category, 4)
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
        return Category_Dynamic_SNG_V3.Dynamic_SNG(categories=category, step=4,
                                                   pruning=True, sample_with_prob=False,
                                                   utility_function='log', utility_function_hyper=0.4,
                                                   momentum=True, gamma=0.9)
    else:
        raise NotImplementedError


M = 10
N = 10
category = [M]*N
# test_function = SumCategoryTestFunction(category)
# ['quad', 'linear', 'exp', 'constant']
# ['index_sum', 'rastrigin', 'rosenbrock ']
epoc_function = 'linear'
func = 'rastrigin'
test_function = EpochSumCategoryTestFunction(category, epoch_func=epoc_function, func=func)
optimizer_name = 'SNG'

# distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
distribution_optimizer = get_optimizer(optimizer_name, category)
runing_times = 500
runing_epochs = 200
save_dir = '/userhome/project/Auto_NAS_V2/experiments/toy_example/'
file_name = '{}_{}_{}_{}_{}_{}.npz'.format(optimizer_name, str(N), str(M), str(runing_epochs),
                                           epoc_function, func)
file_name = os.path.join(save_dir, file_name)
record = {
    'objective': np.zeros([runing_times, runing_epochs]) - 1,
    'l2_distance': np.zeros([runing_times, runing_epochs]) - 1,
}
running_time_interval = np.zeros(runing_times)
for i in tqdm.tqdm(range(runing_times)):
    start_time = time.time()
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
    end_time = time.time()
    running_time_interval[i] = end_time - start_time
    test_function.re_new()
    del distribution_optimizer
    distribution_optimizer = get_optimizer(optimizer_name, category)
# mean_obj = np.mean(record['objective'], axis=0)
# mean_distance = np.mean(record['l2_distance'], axis=0)
np.savez(file_name, record['l2_distance'], running_time_interval)
