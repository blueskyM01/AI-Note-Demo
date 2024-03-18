import os
import xml.dom.minidom
 
# xml_file = '/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/generate_train_label/ttest.xml'
 
#     # print(xml_file)
# #获取 xml 文档对象
# domTree = parse(xml_file)
# #获得根节点
# rootNode = domTree.documentElement
 
# # print('显示xml文档内容')
# # print(rootNode.toxml())
# # print('*'*10)
 
# #判断根节点是否有属性
# if rootNode.hasAttribute('type'):
#     print('根节点的的type属性为：',rootNode.getAttribute('type'))
# else:
#     print('根节点没有属性')
 
# book=rootNode.getElementsByTagName('book')

# print('有%d个book节点'%len(book))
 
# print(rootNode.getElementsByTagName('title')[0].childNodes[0].nodeValue)
# print("rootNode.getElementsByTagName('title')[0] 是获取文档第一个title元素")
# print("childNodes[0]是‘title'元素的第一个子元素，也就是文本节点")
# print("nodeValue为获取节点的值")
 
# print('*'*100)
 
# for i in range(len(book)):
#     print('\n显示第%d个book节点的内容:'%(i+1))
#     print(rootNode.getElementsByTagName('book')[i].toxml())
#     print()
#     if rootNode.getElementsByTagName('book')[i].hasAttribute('category'):
#             print('book节点的属性是：',book[i].getAttribute('category'))
        
#     print('title的值：',book[i].getElementsByTagName('title')[0].childNodes[0].data)
#     #从根节点写是print(rootNode.getElementsByTagName('book')[1].getElementsByTagName('title')[0].childNodes[0].data)
    
#     print('author的值：',book[i].getElementsByTagName('author')[0].childNodes[0].data)
#     #或者用childNodes[0].nodeValue
#     # print('author的值：', book[i].getElementsByTagName('author')[0].childNodes[0].nodeValue)
    
#     print('pageNumber的值：',book[i].getElementsByTagName('pageNumber')[0].childNodes[0].data)
#     print('*' * 10)
 


#获取 xml 文档对象
domTree = xml.dom.minidom.parse('/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/generate_train_label/annotations.xml')
#获得根节点
rootNode = domTree.documentElement
# print('显示xml文档内容')
# print(rootNode.toxml())

images=rootNode.getElementsByTagName('image')
num_images = len(images)

for i in range(num_images):
    sub_noe = images[i]
    image_name = sub_noe.getAttribute('name')
    points = sub_noe.getElementsByTagName('points')
    polygons = sub_noe.getElementsByTagName('polygon')
    
    num_points = len(points)
    num_polygons = len(polygons)
    
    if(num_points == 0) and (num_polygons == 0):
        print("********************** No label! **********************")
    else:
        print('********************** ', image_name, ' **********************')
        if num_points != 0:
            for j in range(num_points):
                index = points[j].getElementsByTagName('attribute')[0].childNodes[0].data
                label = points[j].getAttribute('label')
                points_xy = points[j].getAttribute('points')
                points_xy = points_xy.split(';')
                for xy in points_xy:
                    x = xy[0]
                    y = xy[1]
                
                print('label:', label)
                print('index:', index)
                print("points: ", points_xy)
            
        if num_polygons != 0:
            for k in range(num_polygons):
                label = polygons[k].getAttribute('label')
                points_xy = polygons[k].getAttribute('points')
                
                print('label:', label)
                print("polygons: ", points_xy)
        
    
print('There are %d images!' % num_images)



