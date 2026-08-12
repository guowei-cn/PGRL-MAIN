import numpy as np
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F

class BadNet(object):
    def __init__(self, trigger_path):
        # with open(trigger_path, "rb") as f:
        #     trigger_ptn = Image.open(f).convert("RGB").resize((size, size))
        self.trigger_ptn = np.load(trigger_path)
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        img[self.trigger_loc] = 0
        poison_img = img + self.trigger_ptn
        poison_img = np.uint8(poison_img)

        return poison_img


class Blend(object):

    def __init__(self, trigger_path, alpha=0.1):
        # with open(trigger_path, "rb") as f:
        #     self.trigger_ptn = Image.open(f).convert("RGB").resize((size, size))
        self.trigger_ptn = np.load(trigger_path)
        self.alpha = alpha

    def __call__(self, img):
        return self.blend_trigger(img)

    def blend_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        # img = Image.fromarray(img)
        # poison_img = Image.blend(img, self.trigger_ptn, self.alpha)
        poison_img = self.alpha * self.trigger_ptn + (1 - self.alpha) * img
        poison_img = np.clip(poison_img, 0, 255)
        poison_img = np.uint8(poison_img)

        return poison_img


class SIG(object):

    def __init__(self, trigger_path, alpha=0.5):
        self.trigger_ptn = np.load(trigger_path)
        self.alpha = alpha

    def __call__(self, img):
        return self.blend_trigger(img)

    # def create_SIG(self, size, delta=20, f=6):
    #     H = size
    #     W = size
    #     C = 3
    #     pattern = np.zeros((H, W, C))
    #     m = W
    #     for i in range(H):
    #         for j in range(W):
    #             pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    #
    #     return pattern

    def blend_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        # img = Image.fromarray(img)
        # trigger_ptn = self.trigger_ptn.resize(img.size)
        # poison_img = Image.blend(img, trigger_ptn, self.alpha)
        poison_img = self.alpha * self.trigger_ptn + (1 - self.alpha) * img
        poison_img = np.clip(poison_img, 0, 255)
        poison_img = np.uint8(poison_img)

        return poison_img


class WaNet(object):

    def __init__(self, trigger_path):
        self.trigger_path = trigger_path
        self.trigger_ptn = torch.load(trigger_path)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        bd_grids = self.trigger_ptn
        data = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        data = F.grid_sample(data, bd_grids, align_corners=True)
        poison_img = data.permute(0, 2, 3, 1).squeeze(0).to(torch.uint8).numpy()
        poison_img = np.uint8(poison_img)
        return poison_img

class WaNet_noisy(object):

    def __init__(self, trigger_path):
        self.trigger_ptn = torch.load(trigger_path)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        bd_grids = self.trigger_ptn
        ins = torch.rand(1, img.shape[0], img.shape[1], 2) * 2 - 1
        grid_temps2 = bd_grids + ins / img.shape[0]
        noisy_grids = torch.clamp(grid_temps2, -1, 1)

        data = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        data = F.grid_sample(data, noisy_grids, align_corners=True)
        poison_img = data.permute(0, 2, 3, 1).squeeze(0).to(torch.uint8).numpy()
        poison_img = np.uint8(poison_img)

        return poison_img

class Benign(object):
    def __init__(self, trigger_path=None):
        a = 1
    def __call__(self, img):
        return img
