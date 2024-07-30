from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import time
import numpy as np
import torch
import sys
import optimizers as optimizers
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, mkdirs, format_model_name


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        save_dir = mkdirs(
            os.path.join('./embeddings', args.dataset, format_model_name(args.agg_type), args.task, args.manifold))
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')), logging.StreamHandler()])
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    if args.task != 'lp':
        print('Error, notice that this file is for LP task')
        return
        # Load data

    data = load_data(args, os.path.join('./data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    print(f'>> Number of node: {args.n_nodes}, number of dim: {args.feat_dim}')  # 41143, 240

    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])
    Model = LPModel
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
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
        train_metrics['loss'].backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print('Epoch: {}, Param: {}, Grad mean: {:.3e}, Grad std: {:.3e}'.format(
        #             epoch, name, param.grad.mean(), param.grad.std()))

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {:.6f}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.6f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
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
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        logging.info(f"Saved model in {save_dir}")
    results_roc.append(best_test_metrics['roc'])
    results_ap.append(best_test_metrics['ap'])


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.task == 'lp'
    # f = open(mkdirs('./results/lp') + '/{}_{}.txt'.format(args.dataset, args.agg_type), 'w')
    results_roc = []
    results_ap = []
    train(args)
