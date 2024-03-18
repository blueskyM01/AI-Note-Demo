import numpy as np
import cv2, os, argparse
import random
from collections import defaultdict

def get_label(label_dir, label_name):
        '''
        读取的".txt"文件的每行存储格式为： [num_gt, image_absolute_path, img_width, img_height, label_index, box_1, label_index, box_2, ..., label_index, box_n]
                                  Box_x format: label_index x_min y_min x_max y_max. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
                                  num_gt：
                                  label_index： is in range [0, class_num - 1].
                                  For example:
                                  2 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
                                  2 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320

        :param label_dir:
        :param label_name:
        :return: lines： 将".txt"文件的每行变成列表， 存储到lines这个大列表中
        '''
        label_path = os.path.join(label_dir, label_name)
        lines = []
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                lines.append(line.rstrip('\n').rstrip(' ').split(' '))
                line = f.readline()
        random.shuffle(lines)
        return lines
    
def decode(lines):
    image_keypoints = defaultdict(list)  # 创建一个字典，值的type是list
    for line in lines:
        image_path = line[0]
        for index in range(1, len(line)):
            elem = line[index].split(',')
            points = []
            for i in range(len(elem) // 3):
                x = int(elem[i*3+0])
                y = int(elem[i*3+1])
                v = int(elem[i*3+2])
                points.append([x, y, v])
            image_keypoints[image_path].append(points)
    return image_keypoints
            # print("{}, {}, {}".format(x, y, v))
    
lines = get_label('/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/generate_train_label', 'person_keypoints_val2017.txt')

image_keypoints = decode(lines)
point_size = 2
thickness = -1
counter = 0
for key in image_keypoints.keys():
    counter += 1
    img = cv2.imread(key)
    keypoints_set = image_keypoints[key]
    for keypoints in keypoints_set:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for keypoint in keypoints:
            x = keypoint[0]
            y = keypoint[1]
            v = keypoint[2]
            if(v == 2):
                show_img = cv2.circle(img, (x, y), point_size, color, thickness)
    # if counter % 100 == 0:
    #     cv2.imwrite('/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/' + str(counter) +'.png', show_img)

