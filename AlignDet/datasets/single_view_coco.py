from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class SingleViewCocoDataset(CocoDataset):
    '''Original coco datasets return a single dict, here we return a list'''
    def __getitem__(self, idx):
        if self.test_mode:
            return [self.prepare_test_img(idx)]
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return [data]