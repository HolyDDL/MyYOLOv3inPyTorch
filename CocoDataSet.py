
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from collections import defaultdict
import os
import time
import numpy as np

class COCOset(Dataset):

    def __init__(self, annroot: str, imgroot,type: str, transform=None) -> None:
        '''
            Parameters:
                annroot: 填到.json文件根目录
                imgroot: 填到图片内容的根目录
                type: 加载tran/val数据集 填'train' 或 'val'
                transform: 对数据进行处理, default: None
        '''
        super().__init__()
        print(f'----- Making {type} Dataset -----')
        self.trans = transform
        self.imgroot = imgroot
        annfile = os.path.join(annroot, fr'instances_{type}2017.json')
        # 自动获得index
        coco = COCO(annfile)
        # 一个图片id  对应各种图片信息包括图片名字
        self.coco = coco
        '''
        coco.imgs 存储大字典 
        {
            id1:{
                'license': 
                'file_name': name.jpg
                'coco_url': url
                'height': 
                'width':
                'date_captured': time
                'flickr_url': url
                'id': id1
            }
            id2: {
                ...
            }
            ...
        }
        '''
        '''
        coco.imgToAnns 存储大字典, 内部每一张图片id下对应一个list存储图像分割信息字典
        {
            id1: [
                {
                    'segmentation': [[]]
                    'area':
                    'iscrowd':
                    'image_id':
                    'bbox': [lfttop_x ,lfttop_y, w, h]
                    'category_id': 
                    'id': 图像分割的这一样本id
                },
                {
                    ...
                },
                ...
            ],
            id2: ...
        }
        '''
        pic_info = self.get_pic_info()
        pic_path = self.get_pic_path()
        self.sort_pic_info, self.sort_pic_path = self.sort_pic(pic_info, pic_path)
    
    def sort_pic(self, pic_info, pic_path):
        '''
            Return:
                以每一个锚框作为样本集的infor
                sort_pic_info: dict: {id: bbox=[5]}
                sort_pic_path: dict: {id: path_str}
        '''
        index = 0
        sort_pic_info = defaultdict(np.ndarray)
        sort_pic_path = defaultdict(str)
        for id in pic_info:
            for bbox in pic_info[id]:
                sort_pic_info[index] = bbox
                sort_pic_path[index] = pic_path[id]
                index += 1
        return sort_pic_info, sort_pic_path
        
    
    def __len__(self):
        return len(self.sort_pic_info)

    def __getitem__(self, index):
        pic = Image.open(self.sort_pic_path[index]).convert('RGB')
        labels = self.sort_pic_info[index]
        if self.trans is not None:
            pic = self.trans(pic)
        return pic, labels

        
    
    def get_pic_path(self) -> defaultdict(str):
        '''
            Return:
                大字典下包含str
                {
                    id1: path,
                    id2:...
                    ...
                }
        '''
        tic = time.time()
        pic_path = defaultdict(str)
        for id in self.coco.imgToAnns:
            img_name = self.coco.imgs[id]['file_name']
            pic_path[id] = os.path.join(self.imgroot, img_name)
        print(f'Using {time.time() - tic:.2f}s to get all the paths of the images')
        return pic_path

       
    
    def get_pic_info(self) -> defaultdict(np.ndarray):
        '''
            得到每一张图片的bbox信息并归一化
            Return:
                大字典下包含ndarray shape = [num_bboxes, 5], 5 = minx+miny+w+h+calss 
                {
                    id1: [[minx, miny, w, h, class], ...] ,
                    id2: ...
                    ...
                }
        '''
        tic = time.time()
        pic_info = defaultdict(np.ndarray)     
        for id in self.coco.imgToAnns:
            pic_wh = np.asarray([self.coco.imgs[id]['width'], self.coco.imgs[id]['height']], dtype=float)
            pic_info[id] = []
            for bbox in self.coco.imgToAnns[id]:
                bbox_coordiate = np.asarray(bbox['bbox'],dtype=float)
                normalization_bbox = np.asarray([bbox_coordiate[0]/pic_wh[0], bbox_coordiate[1]/pic_wh[1], 
                                                bbox_coordiate[2]/pic_wh[0], bbox_coordiate[3]/pic_wh[1]], dtype=float)
                bbox_category = np.asarray([bbox['category_id']], dtype=float)
                bbox_info = np.concatenate((normalization_bbox, bbox_category))
                pic_info[id].append(bbox_info)
            pic_info[id] = np.array(pic_info[id])
        print(f'Using {time.time() - tic:.2f}s to get normalizated bboxes')
        return pic_info







if __name__ == '__main__':
    coco = COCOset(r"annotations_trainval2017",'val2017', 'val')