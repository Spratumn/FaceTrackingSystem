import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ...utils.box_utils import match, log_sum_exp
from ...data import cfg
GPU = cfg['gpu_train']


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        # num_classes=2
        # overlap_thresh=0.35
        # prior_for_matching=True
        # bkg_label=0
        # neg_mining=True
        # neg_pos=7
        # neg_overlap=0.35
        # encode_target=False
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        # loc_data: [batch_size, num_priors, 4]
        # conf_data:[batch_size, num_priors, 2]

        priors = priors
        num = loc_data.size(0)  # batch_size
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # get ground truth for every face
        for idx in range(num):
            truths = targets[idx][:, :-1].data  # face local:[[x1,y1,w1,h1],[x2,y2,w2,h2]...]
            labels = targets[idx][:, -1].data  # label->confidence
            defaults = priors.data  # priors boxes local:[batch_size, [x,y,w,h]]
            # threshold:0.35
            # variance:[0.1,0.2]
            # idx:0 or 1 or ... or 63,which image
            # loc_t:[batch_size, num_priors, 4]
            # conf_t:[batch_size, num_priors]
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0  # [batch_size, num_priors]
        # [[False,False,False,False,True,False,....],
        #  [False,False,False,False,False,False,....],...

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [batch_size, num_priors, 4]
        # unsqueeze(n): 增加第n个维度
        # expand_as(loc_data): 扩展尺度大小与loc_data一致

        # positive predicted sample(prior)s location
        # predicted results of offsets between predicted bboxes and prior boxes
        loc_p = loc_data[pos_idx].view(-1, 4)  # [num of True, 4] num of matched prior boxes

        # positive sample(prior)s location
        # target of offsets between ground truth bboxes and prior boxes
        # because the prior boxes are fixed,so we need to use the same indices(pos_idx)
        loc_t = loc_t[pos_idx].view(-1, 4)  # [num of True, 4]

        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # batch_conf:[batch_size*num_priors, 2]

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # log_sum_exp(batch_conf): log(softmax(batch_conf))
        # loss_c: [batch_size*num_priors, 1]

        # Hard Negative Mining
        # choose the small loss to learn
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        # loss_c: all pos boxes -> 0
        loss_c = loss_c.view(num, -1)  # [batch_size, num_priors]
        _, loss_idx = loss_c.sort(1, descending=True)
        # loss_idx: [batch_size, num_priors]
        # _:sorted matrix
        _, idx_rank = loss_idx.sort(1)
        # sort the loss_idx under ascending order
        # idx_rank: [batch_size, num_priors]
        # eg.:
        # loss:       6.5, 0.5, 2.1, 16.8, 7.9
        # indices:    0,   1,   2,   3,   4
        # sorted idx: 3(0),4(1),0(2),2(3),1(4)
        # sorted rank:2,   4,   3,   0,   1
        # choose loss:6.5, 0.5, 2.1

        num_pos = pos.long().sum(1, keepdim=True)
        # num_pos: for each image how many matched prior boxes
        # num_pos: [batch_size, 1]
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # clamp:clamp all elements in input into the range [min,max]
        # self.negpos_ratio*num_pos: 7 * each element in num_pos, [batch_size, 1]
        # num_neg: elements cannot be over the num_priors-1
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # [batch_size, num_priors,2]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # [batch_size, num_priors,2]

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)  # this is the target
        # .gt(0): if (pos_idx+neg_idx) > 0,then that element is True
        # conf_data[(pos_idx+neg_idx).gt(0)]: grab the loss where there losses are True,[18080]
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
