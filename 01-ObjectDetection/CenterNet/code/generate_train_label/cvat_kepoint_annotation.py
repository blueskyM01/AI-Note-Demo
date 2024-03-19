import os, json, argparse
import xml.dom.minidom
from collections import defaultdict
 
'''
1、训练标签
    dict={'image1_path': [[x1,y1,cls,index], [x2,y2,cls,index], ...], 'image2_path': [[x1,y1,cls,index], [x2,y2,cls,index], ...]}

    Description：
    key: image_path;
    value: list，where, the list container several sub lists. The elements of sub list are x , y, cls and index. Note that 1) cls start from 0，if there are 10 classes in dataset，the cls is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9; 2) index only avaliable in container corener point detection. Or use -1 to respresent the index

    For example (train.json): 
    {"train01/image_0000000432.jpeg": [["4", "2", 0, "0"]], 
     "train01/image_0000000523.jpeg": [["4", "2", 0, "1"], ["2", "7", 0, "0"], ["1", "9", 0, "0"], ["3", "0", 0, "0"], ["2", "4", 0, "0"], ["2", "2", 0, "0"], ["2", "7", 0, "0"], ["5", "1", 0, "0"], ["5", "5", 0, "0"], ["5", "3", 0, "0"], ["4", "3", 0, "0"], ["4", "4", 0, "0"], ["4", "8", 0, "0"], ["4", "8", 0, "0"], ["3", "9", 0, "0"], ["2", "7", 0, "0"], ["2", "1", 0, "0"]], 
     "train01/image_0000000524.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000525.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000526.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000527.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000528.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000529.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000530.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000531.jpeg": [["4", "2", 0, "1"]]}

2、类别标签 (`.txt` file)，如果是10个类
cls0
cls1
cls2
cls3
cls4
cls5
cls6
cls7
cls8
cls9

For example (class.txt):
aerop
bicyc
bird
boat
bottl
bus
car
cat
chair
cow
'''
 
class zpmc_GenerateTrainLabel:
    def __init__(self, dataset_dir, ann_dir, ann_name, label_save_dir, label_save_name, class_names):
        self.dataset_dir = dataset_dir
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        #获取 xml 文档对象
        self.domTree = xml.dom.minidom.parse(os.path.join(self.ann_dir, self.ann_name))
        #获得根节点
        self.rootNode = self.domTree.documentElement
        self.name_cat = self.generate_classes_label()
        self.generate_ann_label()
        
        # print('显示xml文档内容')
        # print(rootNode.toxml())
        
    def generate_classes_label(self):
        classes = self.rootNode.getElementsByTagName('meta')[0].getElementsByTagName('task')[0].getElementsByTagName('labels')[0].getElementsByTagName('label')
        num_classes = len(classes)
        name_cat = {}
        for i in range(num_classes):
            class_name = classes[i].getElementsByTagName('name')[0].childNodes[0].data
            name_cat[class_name] = i


        # save ".name" file
        f_name = open(os.path.join(self.label_save_dir, self.class_names), 'w')
        for key in name_cat.keys():
            f_name.write(key + '\n')
        f_name.close()
        return name_cat
    
    def generate_ann_label(self):
        images = self.rootNode.getElementsByTagName('image')
        num_images = len(images)
        
        anns = defaultdict(list)  # 创建一个字典，值的type是list

        for i in range(num_images):
            sub_noe = images[i]
            image_name = sub_noe.getAttribute('name').split('/')[-1]
            image_path = os.path.join(self.dataset_dir, image_name)
            points = sub_noe.getElementsByTagName('points')
            polygons = sub_noe.getElementsByTagName('polygon')
            
            num_points = len(points)
            num_polygons = len(polygons)
            
            if(num_points == 0) and (num_polygons == 0):
                print("********************** '{}' No label! **********************".format(image_path))
                
            else:
                print('image_path: {}'.format(image_path))
                elem = [] # [x, y, cls, index]
                if num_points != 0:
                    for j in range(num_points):
                        index = points[j].getElementsByTagName('attribute')[0].childNodes[0].data
                        label = points[j].getAttribute('label')
                        points_xy = points[j].getAttribute('points')
                        points_xy = points_xy.split(';')
                        for xy in points_xy:
                            x = int(xy[0])
                            y = int(xy[1])
                            anns[image_path].append([x, y, self.name_cat[label], int(index)])
                    
                if num_polygons != 0:
                    for k in range(num_polygons):
                        label = polygons[k].getAttribute('label')
                        points_xy = polygons[k].getAttribute('points')
                        
                        print('label:', label)
                        print("polygons: ", points_xy)
                
        with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
            json.dump(anns, f)    
        print('There are %d images!' % num_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default='/home/ntueee/CVAT_Annotation_Store/train01', type=str, help="the dir of image dataset")
    # parser.add_argument("--dataset_name", default='COCO', type=str, help="the name of image dataset")
    parser.add_argument("--ann_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='annotations.xml', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='train.json', type=str, help="the name of saving generate label")
    parser.add_argument("--class_names", default='classes.txt', type=str, help="the name to classes")
    cfg = parser.parse_args()    
    zpmc_GenerateTrainLabel(dataset_dir=cfg.dataset_dir, ann_dir=cfg.ann_dir, ann_name=cfg.ann_name, label_save_dir=cfg.label_save_dir, label_save_name=cfg.label_save_name, class_names=cfg.class_names)