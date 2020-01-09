""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config.config import SearchConfig
import utils.utils as utils
from model.darts_cnn import SelectSearchCNN
from model.proxyless_cnn import SuperProxylessNASNets
from datasets import get_data
from search_algorithm import Category_MDENAS, Category_DDPNAS, Category_SNG, Category_ASNG, \
    Category_Dynamic_ASNG, Category_Dynamic_SNG, Category_Dynamic_SNG_V3
from utils import genotypes
import random
import pdb


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(logdir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "logger.log"))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")
    logger.info("Torch version is: {}".format(torch.__version__))

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # torch.backends.cudnn.benchmark = True
    if config.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = get_data.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False,
        image_size=config.image_size)
    minimum_image_size = 4 if config.search_space == 'darts' else int(np.prod(np.array(config.stride_stages))*2)
    assert input_size >= minimum_image_size, "input image too small!!"

    # init model and net crit
    net_crit = nn.CrossEntropyLoss().to(device)
    if config.search_space == 'darts':
        model = SelectSearchCNN(input_channels, config.init_channels, n_classes,
                                config.layers, config.n_nodes, net_crit)
        total_edges = sum(list(range(2, config.n_nodes + 2))) * 2
        num_ops = len(genotypes.PRIMITIVES)
    elif config.search_space == 'proxyless_nas':
        model = SuperProxylessNASNets(
            width_stages=config.width_stages, n_cell_stages=config.n_cell_stages, stride_stages=config.stride_stages,
            conv_candidates=config.conv_candidates, n_classes=n_classes,
            width_mult=config.width_mult,
            bn_param=(config.bn_momentum, config.bn_eps), dropout_rate=config.dropout,
            criterion=net_crit
        )
        total_edges = len(model.blocks)
        num_ops = len(config.conv_candidates) + 1
    else:
        raise NotImplementedError
    model = model.to(device)
    # weights optimizer
    w_optim = torch.optim.SGD(model.weight_parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // config.datset_split
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    # using step learning rate instead
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     w_optim, int(config.epochs/num_ops) + 1, eta_min=config.w_lr_min)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
         w_optim, step_size=config.w_lr_step, gamma=config.w_lr_gamma)

    # init optimizer
    if config.name == 'MDENAS':
        distribution_optimizer = Category_MDENAS.CategoricalMDENAS(
            [num_ops]*total_edges, learning_rate=config.theta_lr)
    elif config.name == 'DDPNAS':
        distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(
            [num_ops]*total_edges, 3)
    elif config.name == 'SNG':
        distribution_optimizer = Category_SNG.SNG(
            [num_ops]*total_edges)
    elif config.name == 'ASNG':
        distribution_optimizer = Category_ASNG.ASNG(
            [num_ops]*total_edges)
    elif config.name == 'dynamic_ASNG':
        distribution_optimizer = Category_Dynamic_ASNG.Dynamic_ASNG(categories=[num_ops]*total_edges,
                                                                    step=3,
                                                                    pruning=True)
    elif config.name == 'dynamic_SNG':
        distribution_optimizer = Category_Dynamic_SNG.Dynamic_SNG(categories=[num_ops]*total_edges,
                                                                    step=3,
                                                                    pruning=True)
    elif config.name == 'dynamic_SNG_V3':
        return Category_Dynamic_SNG_V3.Dynamic_SNG(categories=[num_ops]*total_edges, step=3,
                                                   pruning=True, sample_with_prob=False,
                                                   utility_function='log', utility_function_hyper=0.4,
                                                   momentum=True, gamma=0.9)
    else:
        raise NotImplementedError
    # training loop
    logger.info("start warm up training")
    for epoch in range(config.warm_up_epochs):
        # lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        # warm up training
        array_sample = [random.sample(list(range(num_ops)), num_ops) for i in range(total_edges)]
        array_sample = np.array(array_sample)
        for i in range(num_ops):
            sample = np.transpose(array_sample[:, i])
            train(train_loader, valid_loader, model, w_optim, lr, epoch, sample)
    logger.info("end warm up training")
    logger.info("start One shot searching")
    best_top1 = 0.
    best_genotype = None
    lr_flag = 1
    for epoch in range(config.epochs):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        if 'dynamic' in config.name:
            if epoch >= lr_flag * config.w_lr_step and len(distribution_optimizer.sample_index) == 0:
                lr_scheduler.step()
                lr_flag += 1
        else:
            if epoch % config.w_lr_step == 0:
                lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        sample = distribution_optimizer.sampling_index()
        # training
        train(train_loader, valid_loader, model, w_optim, lr, epoch, sample)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step, sample)
        # information recoder
        distribution_optimizer.record_information(sample, top1)
        distribution_optimizer.update()
        # log
        # genotype
        genotype = model.genotype(distribution_optimizer.p_model.theta)
        logger.info("genotype: {}".format(genotype))
        logger.info("The learning rate is: {}".format(lr))
        # logger.info("the theta is = {}".format(distribution_optimizer.p_model.theta))

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
            logger.info("Current best genotype is: {}".format(genotype))
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Training is done, saving the probability")
    np.save(os.path.join(config.path, 'probability.npy'), distribution_optimizer.p_model.theta)


def train(train_loader, valid_loader, model, w_optim, lr, epoch, sample):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device), trn_y.to(device)
        val_X, val_y = val_X.to(device), val_y.to(device)
        N = trn_X.size(0)
        w_optim.zero_grad()
        logits = model(trn_X, sample)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step, sample):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device), y.to(device)
            N = X.size(0)

            logits = model(X, sample)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
