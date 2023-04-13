import torch
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image, ImageFilter
from torchvision import transforms as _transforms

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmcv.utils import build_from_cfg
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, results):
        results['img'] = self.trans(results['img'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class RandomGrayscale(object):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, results):
        num_output_channels = F.get_image_num_channels(results['img'])
        if torch.rand(1) < self.p:
            results['img'] = F.rgb_to_grayscale(results['img'], num_output_channels=num_output_channels)
            return results
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


@PIPELINES.register_module()
class GaussianBlur(object):
    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        device = results['img'].device
        # (C, H, W) -> (H, W, C)
        img = Image.fromarray(results['img'].numpy().astype(np.uint8).transpose(1,2,0))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        results['img'] = torch.from_numpy(np.asarray(img).transpose(2,0,1)).to(device)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    def __init__(self, threshold=128, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'

        self.threshold = threshold
        self.prob = p

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        results['img'] = F.solarize(results['img'], self.threshold)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class TensorNormalize(object):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, results):
        # changd dtype from uint8 into float32
        img = results['img'].to(dtype=torch.float32)
        results['img'] = F.normalize(img, self.mean, self.std, self.inplace)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


@PIPELINES.register_module()
class ImgToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # 连续存储
            img = np.ascontiguousarray(img)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class SelfSupFormatBundle:
    """SelfSup formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            results['img'] = DC(img, padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'