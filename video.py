#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
     
yolo = YOLO()
# 调用摄像头
capture=cv2.VideoCapture('video/coin7.mp4') 
#capture=cv2.VideoCapture(0)
fps = 0.0
while(capture.isOpened):
    t1 = time.time()
    # 读取某一帧
    ret,frame=capture.read()
    
    if not ret:
        break
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame,text,classes=yolo.detect_image(frame)
    frame = np.array(frame)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("video",frame)
    if cv2.waitKey(30) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()