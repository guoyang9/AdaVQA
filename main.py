import os
import argparse
import json, bcolz
import numpy as np
from tqdm import tqdm
from pprint import pprint
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import utils.config as config
import utils.data as data
import utils.utils as utils
import model.model as model


def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ, idxs, accs = [], [], []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for idx, v, q, a, b, m, q_len in loader:
        v = v.cuda()
        a = a.cuda()
        b = b.cuda()
        q = q.cuda()
        m = m.cuda()
        q_len = q_len.cuda()

        out, v_att = net(v, b, q, q_len)
        if has_answers:
            if config.use_cos:
                out = config.scale * (out - (1 - m))
                # out = config.scale * out

            nll = -F.log_softmax(out, dim=1)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out, a).cpu()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                                acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = torch.cat(answ, dim=0).numpy()
        if has_answers:
            accs = torch.cat(accs, dim=0).numpy()
        else:
            accs = []
        idxs = torch.cat(idxs, dim=0).numpy()
        return answ, accs, idxs


def saved_for_test(test_loader, result, epoch=None):
    """ in test mode, save a results file in the format accepted by the submission server. """
    answer_index_to_string = {a: s for s, a in test_loader.dataset.answer_to_index.items()}
    results = []
    for answer, index in zip(result[0], result[2]):
        answer = answer_index_to_string[answer.item()]
        qid = test_loader.dataset.question_ids[index]
        entry = {
            'question_id': qid,
            'answer': answer,
        }
        results.append(entry)
    result_file = 'vqa_{}_{}_{}_{}_{}_results.json'.format(
        config.task, config.dataset, config.test_split, config.version, epoch)
    with open(result_file, 'w') as fd:
        json.dump(results, fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='saved and resumed file name')
    parser.add_argument('--resume', action='store_true', help='resumed flag')
    parser.add_argument('--test', dest='test_only', action='store_true')
    parser.add_argument('--gpu', default='0', help='the chosen gpu id')
    args = parser.parse_args()

    print(args)
    print_keys = ['cp_data', 'version', 'train_set', 'use_cos', 'entropy', 'scale']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    cudnn.deterministic = True

    ########################################## ARGUMENT SETTING  ########################################
    if args.test_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError('Resuming requires file name!')
    name = args.name if args.name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.resume:
        target_name = name
        logs = torch.load(target_name)
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        data.preloaded_vocab = logs['vocab']
    else: 
        target_name = os.path.join('logs', '{}'.format(name))
    if not args.test_only:
        print('will save to {}'.format(target_name))

    ######################################### DATASET PREPARATION #######################################
    if config.train_set == 'train':
        val_loader = data.get_loader(val=True)
        if not args.test_only:
            train_loader = data.get_loader(train=True)
    elif args.test_only:
        val_loader = data.get_loader(test=True)
    else:
        train_loader = data.get_loader(train=True, val=True)
        val_loader = data.get_loader(test=True)

    ########################################## MODEL PREPARATION ########################################
    if config.pretrained_model == 'glove':
        embedding = bcolz.open(config.glove_path_filtered)[:] 
    else:
        embedding = len(val_loader.dataset.token_to_index)
    net = model.Net(embeddings=embedding)
    net = nn.DataParallel(net).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    decay_step = 25000 if config.version == 'v1' else 50000
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / decay_step))

    acc_val_best = 0.0
    start_epoch = 0
    if args.resume:
        net.load_state_dict(logs['model_state'])
        optimizer.load_state_dict(logs['optim_state'])
        scheduler.load_state_dict(logs['scheduler_state'])
        start_epoch = logs['epoch']
        acc_val_best = logs['acc_val_best']

    tracker = utils.Tracker()

    r = np.zeros(3)
    for i in range(start_epoch, config.epochs):
        if not args.test_only:
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
        if not (config.train_set == 'train+val' and i in range(config.epochs-5)):
            r = run(net, val_loader, optimizer, scheduler, tracker, train=False, 
                    prefix='val', epoch=i, has_answers=(config.train_set == 'train'))

        if not args.test_only:
            results = {
                'epoch': i,
                'name': name,
                'model_state': net.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2]
                },
                'vocab': val_loader.dataset.vocab,
            }
            if config.train_set == 'train' and r[1].mean() > acc_val_best:
                acc_val_best = r[1].mean()
                results['acc_val_best'] = acc_val_best
                torch.save(results, target_name+'.pth')
            if config.train_set == 'train+val':
                torch.save(results, target_name+'.pth')
                if i in range(config.epochs-5, config.epochs):
                    saved_for_test(val_loader, r, i)
                    torch.save(results, target_name+'{}.pth'.format(i))
                
        else:
            saved_for_test(val_loader, r)
            break
    print("best validation accuracy is {:.2f}".format(100 * acc_val_best))


if __name__ == '__main__':
    main()
