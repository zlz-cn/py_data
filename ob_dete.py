from imageai.Detection import ObjectDetection
import os
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image


def getfile():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def getob():
    #计时
    start = time.time()
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    #载入已训练好的文件
    detector.setModelPath( os.path.join(execution_path , "D:/black/data/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    #将检测后的结果保存为新图片
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , getfile()), output_image_path=os.path.join(execution_path , "D:/black/pic/peopleresult.jpg"))
    #结束计时
    end = time.time()
    for eachObject in detections:
        print(eachObject["name"] ," : " ,eachObject["percentage_probability"] , " : ", eachObject["box_points"] )  ##预测物体名:预测概率:物体两点坐标（左上，右下）
        print("--------------------------------")
    print("\ncost time:",end-start)
    image=Image.open("D:/black/pic/peopleresult.jpg")
    image.show()
#getob()