# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np


import torch
# import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR, LeNetForMNIST
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='LROnMnist')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--stale-threshold', type=int, default=0)
parser.add_argument('--ratio', type=float, default=5)
parser.add_argument('--isCompensate', type=bool, default=False)
parser.add_argument('--file-name', type=str, default='')

# for compensation
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--add', type=int, default=0)


args = parser.parse_args()


# select gradient changes > threshold
def get_upload(g_remain, g_new, ratio, isCompensate, g_threshold):
    g_change = []
    g_change_merge = torch.empty(0)
    for idx, g_layer in enumerate(g_new):
        g_change_layer = g_layer - g_remain[idx]
        g_change_layer = torch.abs(g_change_layer)
        g_change.append(g_change_layer)

        g_change_layer_reshape = g_change_layer.reshape(torch.numel(g_change_layer))
        g_change_merge = torch.cat((g_change_merge, g_change_layer_reshape), dim=0)  # merge two vectors into one vector

    # threshold
    param_num = torch.numel(g_change_merge)
    threshold = ratio * g_threshold / np.sqrt(param_num)

    g_change_new = []
    non_upload_num = 0
    for idx, g_layer in enumerate(g_change):
        mask = g_layer < threshold
        non_upload_num += torch.sum(mask)

        g_change_tmp = g_new[idx] - g_remain[idx]
        g_change_tmp[mask] = 0.0
        g_change_new.append(g_change_tmp)
        
        g_remain[idx] += g_change_tmp
    return g_remain, g_change_new, (param_num-int(non_upload_num))/param_num

# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, iterations_epoch):
    workers_num = len(workers)
    print('Model recved successfully!')
    optimizers_list = []
    for i in workers:
        if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
            optimizer = MySGD(models[i].parameters(), lr=0.1)
        else:
            optimizer = MySGD(models[i].parameters(), lr=0.01)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 500
    else:
        decay_period = 1000000

    print('Begin!')

    global_g = [torch.zeros_like(param.data) for param in model.parameters()]

    # store (train loss, energy, iterations)
    trainloss_file = './trainloss' + args.model + '_' + args.file_name + '_ec.txt' # .txt file name
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i-1]))

    epoch_train_loss = 0.0
    global_clock = 0
    g_remain_list = []
    ratio = args.ratio
    lr = args.lr  # 'MnistCNN', 'AlexNet', 'ResNet18OnCifar10' lr = 0.1 ; else lr = 0.01
    threshold = 0.

    # compensation
    h_last_list = []  # h_t
    h_remain_list = []  # h_t - 1
    alpha = args.alpha
    beta = args.beta
    print(alpha, " and ", beta)
    for iteration in range(args.epochs * iterations_epoch):
        iteration_loss = 0.0

        g_list = []
        g_change_average = [torch.zeros_like(param.data) for param in models[0].parameters()]
        global_clock += 1
        for i in workers:
            try:
                data, target = next(train_data_iter_list[i-1])
            except StopIteration:
                train_data_iter_list[i-1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i-1])
            data, target = Variable(data), Variable(target)
            optimizers_list[i-1].zero_grad()
            output = models[i](data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizers_list[i-1].get_delta_w()
            # last_delta_ws = optimizers_list[i - 1].get_delta_w()
            g_list.append(delta_ws)
            iteration_loss += loss.data.item()/workers_num

            if global_clock == 1:
                g_remain = [torch.zeros_like(g_layer)+g_layer for g_layer in delta_ws]
                g_remain_list.append(g_remain)

                h_remain = [torch.zeros_like(g_layer) for g_layer in delta_ws]
                h_remain_list.append(h_remain)

                h_last = [torch.zeros_like(g_layer) for g_layer in delta_ws]
                h_last_list.append(h_last)

                last_delta_ws = optimizers_list[i - 1].get_delta_w()
                # synchronous update
                # the gradient change in the first iteration is gradient itself
                for g_change_layer_idx, g_change_layer in enumerate(g_change_average):
                    g_change_layer.data += delta_ws[g_change_layer_idx].data/workers_num
                sparsification_ratio = 1.0
            else:
                # update h
                h_last = [torch.zeros_like(g_layer) for g_layer in delta_ws]
                h_remain = h_last_list[i - 1]
                for idx, g_layer in enumerate(delta_ws):
                    h_last[idx] = h_remain[idx] * beta
                    if args.add == 1:            # default add = 0
                        h_last[idx] += (1/lr) * (last_delta_ws[idx] - g_remain[idx])
                    else:
                        h_last[idx] -= (1/lr) * (last_delta_ws[idx] - g_remain[idx])
                h_remain_list[i - 1] = h_remain
                h_last_list[i - 1] = h_last

                last_delta_ws = optimizers_list[i - 1].get_delta_w()

                # new_delta_ws = [torch.zeros_like(g_layer)+g_layer for g_layer in delta_ws]
                new_delta_ws = optimizers_list[i-1].get_delta_w()
                for idx, g_layer in enumerate(delta_ws):
                    # print(new_delta_ws[idx], " and ", alpha * (h_last_list[i-1][idx] - h_remain_list[i-1][idx]))
                    new_delta_ws[idx] += lr * alpha * (h_last_list[i-1][idx] - h_remain_list[i-1][idx])
                print(ratio)
                g_remain, g_large_change, sparsification_ratio = get_upload(g_remain_list[i-1],new_delta_ws, ratio, args.isCompensate, threshold)
                g_remain_list[i-1] = g_remain
                # synchronous update
                for g_change_layer_idx, g_change_layer in enumerate(g_change_average):
                    g_change_layer.data += g_large_change[g_change_layer_idx].data/workers_num
                

            
            



        # 同步操作
        g_quare_sum = 0.0   # for threshold

        for p_idx, param in enumerate(models[0].parameters()):
            global_g[p_idx].data += g_change_average[p_idx].data
            param.data -= global_g[p_idx].data
            for w in workers:
                list(models[w].parameters())[p_idx].data =  param.data + torch.zeros_like(param.data)

            g_quare_sum += torch.sum(global_g[p_idx].data * global_g[p_idx].data)

        g_quare_sum = torch.sqrt(g_quare_sum)
        threshold = g_quare_sum.data.item()

        epoch_train_loss += iteration_loss
        epoch = int(iteration / iterations_epoch)
        # print('Epoch {}, Loss:{}'.format(epoch, loss.data.item()))
        if (iteration+1) % iterations_epoch == 0:
            # 训练结束后进行test
            test_loss, test_acc = test_model(0, model, test_data, criterion=criterion)
            f_trainloss.write(str(args.this_rank) +
                              "\t" + str(epoch_train_loss / float(iterations_epoch)) +
                              "\t" + str(iteration_loss) +
                              "\t" + str(0) +
                              "\t" + str(epoch) +
                              "\t" + str(0) +
                              "\t" + str(iteration) +
                              "\t" + str(sparsification_ratio) +        # time
                              "\t" + str(global_clock) +        # time
                              "\t" + str(test_loss) +  # test_loss
                              "\t" + str(test_acc) +  # test_acc
                              '\n')
            f_trainloss.flush()
            epoch_train_loss = 0.0
            # 在指定epochs (iterations) 减少缩放因子
            if (epoch + 1) in [0, 1000]:
                ratio = ratio * 0.1
                print('--------------------------------')
                print(ratio)

            for i in workers:
                models[i].train()
                if (epoch + 1) % decay_period == 0:
                    for param_group in optimizers_list[i - 1].param_groups:
                        param_group['lr'] *= 0.1
                        print('LR Decreased! Now: {}'.format(param_group['lr']))

    f_trainloss.close()

def init_processes(workers,
                   models, save_path,
                   train_dataset_list, test_dataset,iterations_epoch,
                   fn, backend='tcp'):
    fn(workers, models, save_path, train_dataset_list, test_dataset, iterations_epoch)


if __name__ == '__main__':

    torch.manual_seed(1)
    workers_num = args.workers
    workers = [v+1 for v in range(workers_num)]
    models = []

    for i in range(workers_num + 1):
        if args.model == 'MnistCNN':
            model = MnistCNN()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LeNet':
            model = LeNetForMNIST()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnMnist':
            model = ResNetOnCifar10.LROnMnist()
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnCifar10':
            model = ResNetOnCifar10.LROnCifar10()
            train_transform, test_transform = get_data_transform('cifar')

            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'AlexNet':

            train_transform, test_transform = get_data_transform('cifar')

            if args.data_name == 'cifar10':
                model = AlexNetForCIFAR()
                train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                                 transform=train_transform)
                test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                                transform=test_transform)
            else:
                model = AlexNetForCIFAR(num_classes=100)
                train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                                  transform=train_transform)
                test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                                 transform=test_transform)
        elif args.model == 'ResNet18OnCifar10':
            model = ResNetOnCifar10.ResNet18()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        elif args.model == 'ResNet34':
            model = models.resnet34(pretrained=False)

            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
            train_dataset = datasets.ImageFolder(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
        models.append(model)
    train_bsz = args.train_bsz
    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    train_data_list = []
    for i in workers:
        train_data_sub = select_dataset(workers, i, train_data, batch_size=train_bsz)
        train_data_list.append(train_data_sub)

    test_bsz = 400
    # 用所有的测试数据测试
    test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle = False)

    iterations_epoch = int(len(train_dataset) / args.train_bsz)

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(workers,
                                                  models, save_path,
                                                  train_data_list, test_data,iterations_epoch,
                                                  run))
    p.start()
    p.join()
