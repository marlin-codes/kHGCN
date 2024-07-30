from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import time
import math
import numpy as np
import torch
import optimizers as optimizers
from config import parser
from models.base_models import NCModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, mkdirs, format_model_name
from config import args


def train(args):
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
        print('>> using double precision')
    if int(args.cuda) >= 0:
        args.device = 'cuda:' + str(args.cuda)
    else:
        args.device = 'cpu'

    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = mkdirs(
                os.path.join('./embeddings', args.dataset, format_model_name(args.agg_type), args.task, args.manifold))
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')), logging.StreamHandler()])
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(args.data_root, args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        print('Error, notice that this file is for nc task')
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info('Manifold:{}'.format(args.manifold))
    print('==' * 20)
    logging.info(str(model))
    print('==' * 20)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
            else:
                if isinstance(val, list) and len(val) == 2 and args.agg_type == 'curv':
                    data[x] = [val[i].to(args.device) for i in range(len(val))]
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        orc_loss = model.compute_metrics_orc(embeddings, data, 'train', threshold=0.0, dataset=args.dataset)
        loss = train_metrics['loss'] + args.beta * orc_loss['loss']

        if math.isnan(loss.item()):
            logging.info('loss is nan, break!!!')
            raise ValueError
            break

        loss.backward()

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {:.4f}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_curvature = model.c
                best_emb = embeddings.cpu()
                if args.save:
                    torch.save(best_emb.detach(), os.path.join(save_dir, 'embeddings.pt'))
                    torch.save(best_curvature.detach(), os.path.join(save_dir, 'layer_curvatures.pt'))
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        torch.save(best_emb.detach(), os.path.join(save_dir, 'embeddings.pt'))
        torch.save(best_curvature.detach(), os.path.join(save_dir, 'layer_curvatures.pt'))
        logging.info(f"Saved model in {save_dir}")
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, '{}_adj.pt'.format(args.agg_type))
            torch.save(model.encoder.att_edge.cpu(), filename)
            print('Dumped attention adj: ' + filename)
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))

    results_acc.append(best_test_metrics['acc'])
    results_f1.append(best_test_metrics['f1'])


if __name__ == '__main__':
    results_acc = []
    results_f1 = []
    # f = open(mkdirs('./results/nc') + '/{}_{}.txt'.format(args.dataset, args.agg_type), 'w')
    train(args)
