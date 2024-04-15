import cv2, os, argparse, json
import numpy as np
import random, copy
from collections import defaultdict


class crop_image:
    def __init__(self, label_path, class_name_path, crop_images_save_dir, crop_images_annotiaon_save_path, crop_num):
        self.label_path = label_path
        self.class_name_path = class_name_path
        self.crop_images_save_dir = crop_images_save_dir
        self.crop_images_annotiaon_save_path = crop_images_annotiaon_save_path
        self.crop_num = crop_num
        
        
        
        self.classes = self.get_classes()
        self.generate_clip()
        
    def get_classes(self):
        classes = []
        with open(self.class_name_path, 'r') as f:
            line = f.readline()
            while line:
                classes.append(line.rstrip('\n'))
                line = f.readline()
        return classes
    
    def get_label(self):
            with open(self.label_path, 'r') as load_f:
                load_dict = json.load(load_f)
            load_f.close()
            # image_names = list(load_dict.keys())
            # random.shuffle(lines)
            return load_dict  
          
    def generate_clip(self):
        load_dict = self.get_label()
        images_path = load_dict.keys()
        counter = 0
        
        new_anns = defaultdict(list)  # 创建一个字典，值的type是list
        
        if not os.path.exists(self.crop_images_save_dir):
            os.makedirs(self.crop_images_save_dir)
        
        for image_path in images_path:
            img = cv2.imread(image_path)
            img_name = image_path.split('/')[-1].split('.')[0]
            h, w, c = img.shape
            anns = load_dict[image_path]
            counter += 1
            for ik, ann in enumerate(anns):
                x = int(ann[0])
                y = int(ann[1])
                cls = int(ann[2])
                index = int(ann[3])
                
                
                for idx in range(self.crop_num):
                    rx = random.randint(50, 210)
                    ry = random.randint(50, 210)
                    x0 = x - rx
                    y0 = y - ry
                    x1 = x0 + 256
                    y1 = y0 + 256
                    
                    # boundaru check
                    if x0 < 0:
                        x0 = 0
                        x1 = x0 + 256
                    if y0 < 0:
                        y0 = 0
                        y1 = y0 + 256
                    if x1 > w -1:
                        x1 = w
                        x0 = w -256
                    if y1 > h -1:
                        y1 = h
                        y0 = h -256
                        
                    clip_image = img[y0:y1, x0:x1, :]
                    new_x = x - x0
                    new_y = y - y0
                    new_cls = cls
                    new_index = index
                    new_img_name = img_name + '_cls_' + str(cls) + '_'+ str(idx) + '.jpg'
                    new_anns[os.path.join(self.crop_images_save_dir, new_img_name)].append([new_x, new_y, new_cls, new_index])
                    copy_anns = copy.deepcopy(anns) 
                    copy_anns.pop(ik)
                    for elel in copy_anns:
                        x_t = int(elel[0])
                        y_t = int(elel[1])
                        cls_t = int(elel[2])
                        index_t = int(elel[3])
                        if x_t > x0 and y_t > y0 and x_t < x1 and y_t < y1:
                            new_anns[os.path.join(self.crop_images_save_dir, new_img_name)].append([x_t - x0, y_t - y0, cls_t, index_t])
                            # print(os.path.join(self.crop_images_save_dir, new_img_name))
                    
                    cv2.imwrite(os.path.join(self.crop_images_save_dir, new_img_name), clip_image)
                    print('Saving {}'.format(os.path.join(self.crop_images_save_dir, new_img_name)))
            
            
        with open(self.crop_images_annotiaon_save_path, "w") as f:
            json.dump(new_anns, f)    

        print('num_orignal images: {}'.format(len(images_path)))
        print('generate {} crop images:'.format(len(new_anns.keys())))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/CellGuideContainerCorner_train01.json', type=str, help=" ")
    parser.add_argument("--class_name_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/CellGuideContainerCorner.txt', type=str, help=" ")
    parser.add_argument("--crop_images_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/Crop_CellGuideContainerCorner_train01', type=str, help=" ")
    parser.add_argument("--crop_images_annotiaon_save_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/Crop_CellGuideContainerCorner_train01.json', type=str, help=" ")
    parser.add_argument("--crop_num", default=10, type=int, help=" ")
    
    cfg = parser.parse_args()  
    crop_image(cfg.label_path, cfg.class_name_path, cfg.crop_images_save_dir, cfg.crop_images_annotiaon_save_path, cfg.crop_num)         
    

