import cv2
import torch
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        
        self.img_idx = []
        self.anno_idx = []
        self.bbox = []
        self.obj_name = []
        train_txt_path = self.root_path + "/ImageSets/存放用于Training 的图片的名称的txt文件"
        self.img_path = self.root + "/JPEGImage/存放.jpg图片的地址"
        self.anno_path = self.root + "/Annotations/存放annotation标注.xml文件的地址"
        train_txt = open(train_txt_path)
        lines = train_txt.readlnes()
        for line in lines:
            name = line.strip().split()[0]
            self.img_idx.append(self.img_path + name+ '.jpg')
            self.ano_idx.append(self.ano_path+ name + '.xml')
				
    def __getitem__(self, item):
        img = cv2.imread(self.img_idx[item])
        height, width, channels = img.shape
        targets = ET.parse(self.ano_idx[item])
        res = [] #存储标注信息 即边框左上角和右下角的四个点的坐标信息和目标的类别标签
        for obj in targets.iter('object'):
            name = obj.find('name').text.lower().strip()
            class_idx = dict_classes[name]
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            obj_bbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt))
                cur_pt = cur_pt/ width if i% 2 ==0 else cur_pt / height # 将坐标做一个线性变换
                obj_bbox.append(cur_pt)
            res.append(obj_bbox)
            res.append(class_idx)
        return img, res
	
    def __length__(self):
        data_length = len(self.img_idx)
        return data_lenth

    

