from yolo import YOLO
from PIL import Image
import tensorflow as tf
import cv2
import os
import numpy as np
import time
import queue

iou_threshold=0.1
img_pred_path='img_2'
img_save_path='img_detected_with2maxpooling'
true_boxes_path='2007_train.txt'


def get_IOU(ypre,ytrue):

    #矩形框宽高
    w1=ytrue[2]-ytrue[0]
    h1=ytrue[3]-ytrue[1]
    w2=ypre[2]-ypre[0]
    h2=ypre[3]-ypre[1]
    
    #交叉区域宽高
    IOU_w=max(0,(min(ytrue[0],ypre[0])+w1+w2-max(ytrue[2],ypre[2])))
    IOU_h=max(0,(min(ytrue[1],ypre[1])+h1+h2-max(ytrue[3],ypre[3])))

    intersect_areas = IOU_w*IOU_h

    true_areas = (ytrue[2]-ytrue[0]) * (ytrue[3]-ytrue[1])
    pred_areas = (ypre[2]-ypre[0]) * (ypre[3]-ypre[1])

    union_areas = pred_areas + true_areas - intersect_areas
    IOU  = intersect_areas/union_areas

    return IOU

def statistics(ypre_list,ytrue_list,iou_threshold):
    ypre_lenth=len(ypre_list)
    ytrue_lenth=len(ytrue_list)
    TP=0

    if ypre_lenth==0:
        return TP,0,ytrue_lenth
    if ytrue_lenth==0:
        return TP,ypre_lenth,0

    #构建iou矩阵
    A=np.zeros([ytrue_lenth,ypre_lenth,2])
    #挨个计算iou，并填入A矩阵
    for i,ypre in enumerate(ypre_list):
        for j,ytrue in enumerate(ytrue_list):
            iou=get_IOU(ypre,ytrue)
            A[j,i,0]=iou
    
    #进行iou比较
    for col in range(ypre_lenth):
        matched=max(A[:,col,0])
        matched_array=np.argwhere(A[:,col,0]==matched)
        if A[matched_array[0][0],col,1]==0 and max(A[matched_array[0][0],:,0])==matched and matched>iou_threshold:
            TP+=1
            #已配对的话，后面的ytrue就不用比了
            for i in range(ypre_lenth):
                A[matched_array[0][0],i,1]=1
            
    FP=ypre_lenth-TP
    FN=ytrue_lenth-TP

    return TP,FP,FN


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO()

#读取真实框信息
with open(true_boxes_path) as f:
    lines=f.readlines()
true_boxes=[]
for each in lines:
    row=each.strip()
    true_boxes.append(row.split(' '))
for each in true_boxes:
    each[0]=os.path.basename(each[0])
#生成ytrue_boxes
ytrue_boxes=[]
for each in true_boxes:
    box_=[]
    for j in each:       
        if ',' in j:
            box=[int(i) for i in j.split(',') if i!='0']
            box_.append(box)
        else:
            box_.append(j)
    ytrue_boxes.append(box_)

TP_sum=FP_sum=FN_sum=0
#传入待预测的图片
imgs_pred=os.listdir(img_pred_path)
for img in imgs_pred:
    image = Image.open(os.path.join(img_pred_path,img))
    #进行预测
    r_image,predict_boxes = yolo.detect_image(image)
    frame = np.array(r_image)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    #进行查找，并画出真实框,否则box为空
    for ytrue_box in ytrue_boxes:
        if img==ytrue_box[0]:
            #画框框
            for box in ytrue_box[1:]:
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color=(0,255,0),thickness=1)
            break
    else:
        ytrue_box=[]

    # print(predict_boxes,ytrue_box[1:])

    #统计预测框与真实框的正确值和误判值
    TP,FP,FN=statistics(predict_boxes,ytrue_box[1:],iou_threshold)
    TP_sum+=TP
    FP_sum+=FP
    FN_sum+=FN
    text='TP=%s,FP=%s,FN=%s' %(TP,FP,FN)
    # print(text)
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(img,frame)
    # cv2.imwrite(os.path.join(img_save_path,img),frame)
    cv2.waitKey(0)
    # time.sleep(1)
    cv2.destroyAllWindows()

#计算识别率和误判率
recog_rate=TP_sum/(TP_sum+FP_sum)
error_rate=(FP_sum+FN_sum)/(TP_sum+FP_sum+FN_sum)
print(TP_sum,FP_sum,FN_sum)
print('精确率', recog_rate,'\n误判率', error_rate)

