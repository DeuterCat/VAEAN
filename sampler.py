import numpy as np
import torch
from torch.utils.data import Sampler



class FewshotSampler(Sampler):
    '''
    this is a sampler for fewshot learning tasks
    as it can sample 
    it returns a batch with way_num x 15+1 images
    
    '''
    def __init__(
        self,
        dataset,
        total_cat,
        episode_num,
        way_num,
        image_num,
    ):
        super(FewshotSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num
        self.total_cat = total_cat

    
    def __iter__(self):
        
        batch = []
        for epidsode in range(self.episode_num):
            # every batch is of size[80, 3, 84, 84]
            chosenidxes = torch.randperm(self.total_cat)[: self.way_num]
            chosenimages = torch.randperm(600)[: self.image_num]
            for c in chosenidxes:
                for i in chosenimages:
                    idx = 600*c+i
                    batch.append(idx)
            batch = torch.stack(batch)
            yield batch
            batch = []



    def __len__(self):
        return self.episode_num
