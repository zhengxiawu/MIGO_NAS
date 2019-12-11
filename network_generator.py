import numpy as np
import itertools
from utils import utils
from utils import genotypes
import os
import copy


def un_pack_list(input_list, n_node=4):
    assert len(input_list) == sum([i + 2 for i in range(n_node)])
    out_list = []
    _pre = 0
    for i in range(n_node):
        out_list.append(input_list[_pre:_pre+i+2])
        _pre = _pre+i+2
    return out_list


def get_best_gene(height_constraint_graph, _skip_connection_selection_list, max_indexes, _prob):
    # get the best
    best_expectation = 0
    best_gene = None
    for graph in height_constraint_graph:
        for _skip_connection_selection in _skip_connection_selection_list:
            expectation = 0
            graph_operation_index = []
            for node_index in range(len(graph)):
                node_operation_index = []
                # _node: record the input node
                _node = graph[node_index]
                left_node_index, right_node_index = (2*node_index), 2*node_index + 1

                # get the node operation index
                if left_node_index in _skip_connection_selection:
                    node_operation_index.append(max_indexes[node_index][_node[0]][0])
                else:
                    node_operation_index.append(max_indexes[node_index][_node[0]][1])

                if right_node_index in _skip_connection_selection:
                    node_operation_index.append(max_indexes[node_index][_node[1]][0])
                else:
                    node_operation_index.append(max_indexes[node_index][_node[1]][1])

                expectation += _prob[node_index][_node[0]][node_operation_index[0]]
                expectation += _prob[node_index][_node[1]][node_operation_index[1]]
                graph_operation_index.append(node_operation_index)

            if expectation > best_expectation:
                best_expectation = expectation
                # generate the gene
                best_gene = genotypes.parse_graph_and_operation(graph, graph_operation_index)
    return best_gene


def get_network(probability, reduce_constrain=True, height_constraint=2, skip_connection_constraint=2):
    max_indexs = []
    for i in probability:
        _prob = copy.deepcopy(i)
        _ = []
        _.append(np.argmax(_prob))
        _prob[_[0]] = np.nan
        _.append(np.nanargmax(_prob))
        if 0 in _:
            _prob[_[1]] = np.nan
            _prob[4:] = np.nan
            _[_.index(0)] = np.nanargmax(_prob)
        _.sort()
        max_indexs.append(_)
    pre_product = []
    this_product = []
    for i in range(4):
        this_combination = itertools.combinations(list(range(i+2)), 2)
        this_combination = list(this_combination)
        if i == 0:
            pre_product = this_combination
            this_product = this_combination
        else:
            this_product = list(itertools.product(pre_product, this_combination))
            pre_product = this_product
    # flat tuple
    this_product = [(a, b, c, d) for ((a, b), c), d in this_product]
    # height constraint
    height_constraint_graph = []
    for graph in this_product:
        height_list = []
        for node in graph:
            left_height = 0 if node[0] in [0, 1] else height_list[node[0]-2]
            right_height = 0 if node[1] in [0, 1] else height_list[node[1]-2]
            node_height = left_height if left_height >= right_height else right_height
            node_height += 1
            height_list.append(node_height)
        graph_height = max(height_list)
        if graph_height <= height_constraint:
            height_constraint_graph.append(graph)
    # unpack the prob
    n_node = 4
    n_edges = sum([i+2 for i in list(range(4))])
    prob_norm = utils.darts_weight_unpack(probability[0:n_edges], n_node)
    prob_reduce = utils.darts_weight_unpack(probability[n_edges:], n_node)
    max_indexes_norm = un_pack_list(max_indexs[0: n_edges])
    max_indexes_reduce = un_pack_list(max_indexs[n_edges: ])
    _skip_connection_selection_list = list(itertools.combinations(list(range(n_node * 2)), skip_connection_constraint))
    # norm cell
    best_norm_gene = get_best_gene(height_constraint_graph, _skip_connection_selection_list,
                                   max_indexes_norm, prob_norm)
    if reduce_constrain:
        best_reduce_gene = get_best_gene(height_constraint_graph,
                                         _skip_connection_selection_list,
                                         max_indexes_reduce, prob_reduce)
    else:
        best_reduce_gene = genotypes.parse_numpy(prob_reduce, k=2)
    concat = range(2, 2 + 4)  # concat all intermediate nodes
    return genotypes.Genotype(normal=best_norm_gene, normal_concat=concat,
                              reduce=best_reduce_gene, reduce_concat=concat)


if __name__ == '__main__':
    dir_name = '/userhome/project/DDPNAS_V2/experiment'
    dirs = os.listdir(dir_name)
    height_constraints = [1, 2, 3, 4]
    skip_connection_constraints = [2, 4, 6, 8]
    reductions = [True, False]
    # get the prob list
    prob_list = []
    for _dir in dirs:
        _path = os.path.join(dir_name, _dir)
        prob_path = os.path.join(_path, 'probability.npy')
        if os.path.exists(prob_path):
            prob = np.load(prob_path)
            prob_list.append(prob)
    for _height_constraint in height_constraints:
        for _skip_connection_constraint in skip_connection_constraints:
            for _reduction in reductions:
                save_text_name = str(_height_constraint) + '_' + str(_skip_connection_constraint) + \
                                 '_' + str(_reduction) + '.txt'
                print(save_text_name)
                file = open(os.path.join(dir_name, 'txt', save_text_name), 'w+')
                for _prob in prob_list:
                    gen = get_network(_prob, reduce_constrain=_reduction,
                                      height_constraint=_height_constraint,
                                      skip_connection_constraint=_skip_connection_constraint)
                    file.write(str(gen) +"\n")
                file.close()