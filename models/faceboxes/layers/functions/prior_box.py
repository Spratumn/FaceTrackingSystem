import torch
from itertools import product as product
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        # self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']  # anchor sizes [32,64,128],[256],[512]
        self.steps = cfg['steps']  # [32,64,128]
        self.clip = cfg['clip']
        self.image_size = image_size  # input image size: 1024x1024 [h,w]
        # [[32,32],[16,16],[8,8]]
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        # iterate 3 scale feature map: 32,16,8
        for k, f in enumerate(self.feature_maps):
            # k=0,f=[32,32] [h,w]
            # k=1,f=[16,16]
            # k=2,f=[8,8]
            min_sizes = self.min_sizes[k]
            # k=0,min_sizes=[32,64,128]
            # k=1,min_sizes=[256]
            # k=2,min_sizes=[512]

            # iterate feature point
            for i, j in product(range(f[0]), range(f[1])):
                # create anchor for every feature point
                for min_size in min_sizes:
                    # get scaled h,w
                    s_kx = min_size / self.image_size[1]  # [32,64,128,256,512]/1024: 0.03125,0.0625,0.125,0.25,0.5
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        # create 16 anchor: 1/4 at top left corner
                        # normalize
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        # create 4 anchor: 1/2 at top left corner
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        # create 1 anchor: 1 at center
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
