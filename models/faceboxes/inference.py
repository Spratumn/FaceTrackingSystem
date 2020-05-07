from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from .data import cfg
from .layers.functions.prior_box import PriorBox
from .models.faceboxes import FaceBoxes
from .utils.nms_wrapper import nms

import cv2

from .utils.box_utils import decode
from .utils.timer import Timer

parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('-m', '--trained_model', default='./weights/FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=True):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def build_net(image_size, device='cpu'):
    device = torch.device(device)
    # fixed weights
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
    net = load_model(net, args.trained_model)
    net.eval()
    print('Finished loading model!')
    # print(net)
    net = net.to(device)
    priorbox = PriorBox(cfg, image_size=image_size)
    priors_out = priorbox.forward()
    priors_out = priors_out.to(device)
    prior_data = priors_out.data
    return net, prior_data


def get_face_boxes(net, prior_data, image, resize, cur_frame_counter, device='cpu'):
    device = torch.device(device)
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError('image path invalid')
        else:
            img_raw = cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        img_raw = image
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    # print(im_height, im_width)
    scale = torch.from_numpy(np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]))
    # scale = torch.Tensor(np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]))
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # _t['forward_pass'].tic()
    loc, conf = net(img)  # forward pass
    # _t['forward_pass'].toc()
    # _t['misc'].tic()

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, args.nms_threshold, force_cpu=True)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    _t['misc'].toc()
    outputs_useful = []
    for b in dets:
        output_traffic = {}
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        (left, right, top, bottom) = (b[0], b[2], b[1], b[3])
        label_str = 'face'
        output_traffic[label_str] = [left, right, top, bottom]
        outputs_useful.append(output_traffic)
    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imwrite('output/' + str(cur_frame_counter) + '.jpg', img_raw)

    return outputs_useful


if __name__ == '__main__':
    net, prior_data = build_net((920, 1380), 'cpu')
    get_face_boxes(net, prior_data, 'XHH_5108.jpg', 0.25, 0)
