import copy
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy
from loss import FedDecorrLoss
from .fedavg import FedAvg

class FedExp(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedExp, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')
        # FedExp
        parser.add_argument('--eps', type=float, default=1e-3,
                            help='epsilon of the FedExp algorithm')
        return parser.parse_args(extra_args)


    def global_aggregation(self, nets_this_round):
        total_data_points = sum([len(self.party2loaders[r].dataset) for r in nets_this_round])
        fed_avg_freqs = [len(self.party2loaders[r].dataset) / total_data_points for r in nets_this_round]

        global_w_init = copy.deepcopy(self.global_net.state_dict())
        global_w_update = copy.deepcopy(self.global_net.state_dict())
        global_w = self.global_net.state_dict()
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w_update[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w_update[key] += net_para[key] * fed_avg_freqs[net_id]

        # calculate eta_g, the server learning rate defined in FedExp
        # ======================================
        # calculate \|\bar{Delta}^{(t)}\|^{2}
        sqr_avg_delta = 0.0
        for key in global_w_update:
            sqr_avg_delta += ((global_w_update[key] - global_w_init[key])**2).sum()

        # calculate \sum_{i}{p_{i}\|\Delta_{i}^{(t)}\|^{2}} for each client
        avg_sqr_delta = 0.0
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            for key in net_para:
                avg_sqr_delta += fed_avg_freqs[net_id] * ((net_para[key] - global_w_init[key])**2).sum()

        eta_g = avg_sqr_delta / (2*(sqr_avg_delta + self.appr_args.eps))
        eta_g = max(1.0, eta_g.item())

        # log eta_g
        self.logger.info('eta_g at current round: %f' % eta_g)
        # ======================================

        for key in global_w:
            global_w[key] = global_w_init[key] + eta_g*(global_w_update[key] - global_w_init[key])
        self.global_net.load_state_dict(global_w)

