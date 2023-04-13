from mmdet.datasets.pipelines import Compose
from mmdet.datasets import PIPELINES
from mmcv.utils import build_from_cfg

class MultiViewAug:
    def __init__(self, num_views, pipelines, keys):
        assert len(num_views) == len(pipelines) == len(keys)

        self.keys = keys
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans

    def __call__(self, results):
        for trans, key in zip(self.trans, self.keys):
            results[key] = trans(results['img'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'