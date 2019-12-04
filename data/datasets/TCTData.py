import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import numpy as np
import sys
import glob

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from ModelCore.structures.bounding_box import BoxList



class tctDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__",
        "hsil",
        "lsil"
    )
    TYPE_TO_NAME = {
        255: "hsil",
        65535: "lsil",
        16711680: "lsil"
    }
    WIDTH, HEIGHT = 512, 512
    def __init__(self, csv_dir, split, transforms=None, use_neg_sample=False):
        cls = tctDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories   = dict(zip(range(len(cls)), cls))
        self.df           = pd.read_csv(csv_dir)
        self.ids          = self.df.file_path.unique()
        self.transforms   = transforms
        self.neg_sample_prob = -1.
        self.split = split
        self.neg_sample_ids = glob.glob('/home/../data/tct/DATA/negative_sample_patches/*.png')
        print("Current data split is {}".format(self.split))

    def __getitem__(self, index):
        #print("{}/{}".format(index, len(self.ids)))
        pick_neg_sample = (np.random.random() <= self.neg_sample_prob)
        if not pick_neg_sample or self.split != 'train':
            img_id = self.ids[index]
            img    = Image.open(img_id).convert("RGB")
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            ind = index % len(self.neg_sample_ids) #np.random.randint(0, len(self.neg_sample_ids), 1)[0]
            img_id = self.neg_sample_ids[ind]
            img = Image.open(img_id).convert("RGB")
            target = torch.tensor([])
            if self.transforms is not None:
                img, _  = self.transforms(img, None)

        return img, target, img_id


    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id  = self.ids[index]

        anno = self._preprocess_annotation(index)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target


    def _preprocess_annotation(self, index):
        boxes = []
        gt_classes = []

        TO_REMOVE = 1

        img_id = self.ids[index]
        anno_df = self.df.loc[self.df.file_path==img_id]
        for ind, row in anno_df.iterrows():
            x1, y1 = int(row.bbox_x), int(row.bbox_y)
            x2, y2 = int(x1 + row.bbox_w), int(y1 + row.bbox_h)
            box = [x1, y1, x2, y2]
            bnbox = tuple(map(lambda x: x - TO_REMOVE, box))
            boxes.append(bnbox)

            cell_type = int(row.type)
            gt_classes.append(self.class_to_ind[self.TYPE_TO_NAME[cell_type]])

        im_info = (self.WIDTH, self.HEIGHT)

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info
        }
        return res



    def get_img_info(self, index):
        return {"height":512, "width":512}

    def map_class_id_to_class_name(self, class_id):
        return tctDataset.CLASSES[class_id]

