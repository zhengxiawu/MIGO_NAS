import os
import sys
from datetime import datetime
import time
import shutil
import glob
import shutil

# walk_dir = sys.argv[1]
#
# print('walk_dir = ' + walk_dir)
#
# print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))


def get_folder_list(walk_dir):
    folder_list = []
    walk_dir = os.path.abspath(walk_dir)
    method_list = os.listdir(walk_dir)
    for method in method_list:
        model_list = os.listdir(os.path.join(walk_dir, method))
        if len(method_list) > 0:
            for model in model_list:
                dataset_list = os.listdir(os.path.join(walk_dir, method, model))
                for dataset in dataset_list:
                    train_name_list = os.listdir(os.path.join(walk_dir, method, model, dataset))
                    for train_name in train_name_list:
                        this_train_folder = os.path.join(walk_dir, method, model, dataset, train_name)
                        folder_list.append(this_train_folder)
    return folder_list


def folder_clean(walk_dir, remove_flag = False):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        time_str = log_lines[-1][0:17]
        time_str = time_str.replace(' ', '_')
        time_str = '2020/'+time_str
        FMT = '%Y/%m/%d_%H:%M:%S_%p'
        time_fmt = datetime.strptime(time_str, FMT)
        now_time = datetime.now()
        time_interval = now_time - time_fmt
        if time_interval.seconds > 60 * 60 * 2:
            if not os.path.isfile(os.path.join(this_train_folder, 'best.pth.tar')):
                print(this_train_folder)
                if remove_flag:
                    shutil.rmtree(this_train_folder)


def read_result(walk_dir):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        if 'Final' in log_lines[-1]:
            if 'dali' in this_train_folder.split('/')[-1]:
                print(this_train_folder)
                print(log_lines[-1])


def get_result(search_str):
    project_experiment_path = '/userhome/project/Auto_NAS_V2/'
    folder_list = glob.glob(project_experiment_path + 'experiments/dynamic_SNG_V3/ofa/cifar10/' + search_str)
    folder_list.sort()
    for i in folder_list:
        print(i)
        file = open(os.path.join(i, 'logger.log'))
        file_lines = file.readlines()
        for j in file_lines[-7:-1]:
            if '600' in j[25:]:
                print(j[25:])


def get_network():
    project_experiment_path = '/userhome/project/Auto_NAS_V2/'
    folder_list = glob.glob(project_experiment_path + 'experiments/dynamic_SNG_V3/ofa/cifar10/*')
    folder_list.sort()
    dst_folder = '/userhome/project/Auto_NAS_V2/experiments/dynamic_SNG_V3/ofa/cifar10/structure'
    for i in folder_list:
        if 'width_multi' in i:
            _path = os.path.join(i, 'network_info')
            name = i.split('/')[-1][0:-25]
            name = 'ofa_cifar10_' + name
            _dst_folder = os.path.join(dst_folder, name)
            k = 1
            _ = _dst_folder
            while os.path.isdir(_):
                _ = _dst_folder + '_' + str(k)
                k += 1
            _dst_folder = _
            if not os.path.isdir(_dst_folder):
                os.mkdir(_dst_folder)
            for j in range(6):
                this_structure = str((j+1)*100) + '.json'
                _network_path = os.path.join(_path, this_structure)
                dst_network_path = os.path.join(_dst_folder, this_structure)
                shutil.copy(_network_path, dst_network_path)


if __name__ == '__main__':
    pass
    get_network()
    # folder_clean(walk_dir, True)
    # read_result(walk_dir)
    # result detect
    # search_space = 'ofa'
    # width_multi = 1.2
    # epoch = 200
    # warm_up_epochs = 0
    # lr = 0.01
    # pruning_step = 3
    # search_str = '{0}__dataset_cifar10_width_multi_{1}_epochs_{2}_data_split_10' \
    #              '_warm_up_epochs_{3}_lr_{4}_pruning_step_{5}*'.format(
    #               str(search_space), str(width_multi), str(epoch), str(warm_up_epochs),
    #               str(lr), str(pruning_step))
    # search_str = '*'
    # get_result(search_str)
    pass

