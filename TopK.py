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
import vgg

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='MnistCNN')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train-bsz', type=int, default=400)
parser.add_argument('--ratio', type=float, default=0.01)
parser.add_argument('--isCompensate', type=bool, default=False)
parser.add_argument('--loops', type=int, default=75)
parser.add_argument('--title', type=str, default='TopK')
args = parser.parse_args()

# select top-k gradient changes
def get_upload(g_remain, g_new, ratio, isCompensate, dev):
    for idx, g_layer in enumerate(g_new):
        g_remain[idx] += g_layer

    g_remain_abs_vector = torch.empty(0).cuda(dev)
    g_remain_abs = []
    for idx, g_layer in enumerate(g_remain):
        g_remain_layer_abs = torch.abs(g_remain[idx])
        g_remain_abs.append(g_remain_layer_abs)
        g_remain_layer_abs_reshape = g_remain_layer_abs.reshape(torch.numel(g_remain_layer_abs))
        g_remain_abs_vector = torch.cat((g_remain_abs_vector, g_remain_layer_abs_reshape),dim=0)  # merge two vectors into one vector

    param_num = torch.numel(g_remain_abs_vector)
    k = int(param_num * ratio)
    k = k if k>0 else 1
    top_k = torch.topk(g_remain_abs_vector, k)
    threshold = top_k[0][k-1].item()

    g_upload = []
    for idx, g_layer in enumerate(g_remain_abs):
        mask = g_layer >= threshold
        g_upload_layer = torch.zeros_like(g_layer).cuda(dev)
        g_upload_layer[mask] += g_remain[idx][mask]
        g_remain[idx][mask] = 0.
        g_upload.append(g_upload_layer)

    return g_remain, g_upload

# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, iterations_epoch):
    dev = torch.device('cuda')
    cpu = torch.device('cpu')

    param_num = 0
    for p in models[0].parameters():
        tmp_p = torch.zeros_like(p)
        param_num += torch.numel(tmp_p)

    models[0] = models[0].cuda(dev)
    for i in workers:
        models[i] = models[i].cuda(dev)

    workers_num = len(workers)
    print('Model recved successfully!')

    compression_num = int(param_num * args.ratio)
    compression_num = compression_num if compression_num > 0 else 1

    optimizers_list = []
    for i in workers:
        optimizer = MySGD(models[i].parameters(), lr=args.lr)
        # if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        # elif args.model in ['VGG11']:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        # else:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    elif args.model in ['Abalone', 'Bodyfat', 'Housing']:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 10000
    else:
        decay_period = 1000000

    print('Begin!')

    # store (train loss, energy, iterations)
    # naming rules: title + model_name + number_of_workers
    trainloss_file = './../result/' + args.title + '_' + args.model + '_' + str(args.workers) + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i-1]))

    global_clock = 0
    g_remain_list = []
    for i in workers:
        g_remain = [torch.zeros_like(param.data) for param in models[i].parameters()]
        g_remain_list.append(g_remain)
    # time_logs = open("./record" + str(rank), 'w')
    for epoch in range(args.epochs):
        iteration_loss = 0.0

        # epoch_train_loss = 0
        g_change_average = [torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()]
        global_clock += 1
        for i in workers:
            try:
                data, target = next(train_data_iter_list[i-1])
            except StopIteration:
                train_data_iter_list[i-1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i-1])
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            optimizers_list[i-1].zero_grad()
            output = models[i](data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizers_list[i-1].get_delta_w()
            iteration_loss += loss.data.item()/workers_num

            g_remain_list[i-1], g_large_change = get_upload(g_remain_list[i-1], delta_ws, args.ratio, args.isCompensate, dev)
            # synchronous update
            for g_change_layer_idx, g_change_layer in enumerate(g_change_average):
                g_change_layer.data += g_large_change[g_change_layer_idx].data/workers_num

        # 同步操作
        for p_idx, param in enumerate(models[0].parameters()):
            param.data -= g_change_average[p_idx].data
            for w in workers:
                list(models[w].parameters())[p_idx].data = param.data

        # epoch_train_loss += iteration_loss
        # epoch = int(iteration / iterations_epoch)
        print('Epoch {}, Loss:{}'.format(epoch, loss.data.item()))
        # if (iteration+1) % iterations_epoch == 0:
        # 训练结束后进行test
        # test_loss, test_acc = test_model(0, model, test_data, criterion=criterion)
        # f_trainloss.write(str(args.this_rank) +
        #                     "\t" + str(iteration_loss) +
        #                     "\t" + str(0) +
        #                     "\t" + str(epoch) +
        #                     "\t" + str(0) +
        #                     "\t" + str(sparsification_ratio) +        # time
        #                     "\t" + str(global_clock) +        # time
        #                     '\n')
        f_trainloss.write(str(epoch) + 
                            '\t' + str(global_clock) +
                            '\t' + str(iteration_loss) + 
                            '\t' + str(args.ratio) + 
                            '\n')
        f_trainloss.flush()
        # epoch_train_loss = 0.0
        # 在指定epochs (iterations) 减少缩放因子
        if (epoch + 1) in [0, 250000]:
            ratio = ratio * 0.1
            print('--------------------------------')
            print(ratio)

        # for i in workers:
        #     models[i].train()
        #     if (epoch + 1) % decay_period == 0:
        #         for param_group in optimizers_list[i - 1].param_groups:
        #             param_group['lr'] *= 0.1
        #             print('LR Decreased! Now: {}'.format(param_group['lr']))

    f_trainloss.close()


def init_processes(workers, models, save_path,
                   train_dataset_list, test_dataset, iterations_epoch,
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
        elif args.model == 'VGG11':
            model = vgg.vgg11()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
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
