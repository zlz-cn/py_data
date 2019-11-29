from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog


def getfile():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def loadImage(path):
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    # 图像的大小在size中是（宽，高）
    # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    data = np.array(data).reshape(height, width)/100
    # 查看原图的话，需要还原数据
    new_im = Image.fromarray(data*100)
    new_im.show()
    return data


def pca(data,k):

    n_samples,n_features = data.shape
    # 求均值
    mean = np.array([np.mean(data[:,i]) for i in range(n_features)])
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵
    matrix_ = np.dot(np.transpose(normal_data),normal_data)
    # 有时会出现复数特征值，导致无法继续计算，这里用了不同的图像，有时候会出现复数特征，但是经过

    eig_val,eig_vec = np.linalg.eig(matrix_)
    eigIndex = np.argsort(eig_val)
    eigVecIndex = eigIndex[:-(k+1):-1]
    feature = eig_vec[:,eigVecIndex]
    new_data = np.dot(normal_data,feature)
    # 将降维后的数据映射回原空间
    rec_data = np.dot(new_data, np.transpose(feature)) + mean
    # print(rec_data)
    # 压缩后的数据也需要乘100还原成RGB值的范围
    newImage = Image.fromarray(np.uint8(rec_data*102))
    newImage.show()
    return rec_data

def error(data,recdata):
    sum1 = 0
    sum2 = 0
    # 计算两幅图像之间的差值矩阵
    D_value = data - recdata
    # 计算两幅图像之间的误差率，即信息丢失率
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2/sum1
    print(sum2, sum1, error)


#date=loadImage("D:/black/data/suns.jpg")
#pca(date,20)

def gatmain():
    data=loadImage(getfile())
    pca = PCA(n_components=10).fit(data)
    # 降维
    x_new = pca.transform(data)
    # 还原降维后的数据到原空间
    recdata = pca.inverse_transform(x_new)
    # 计算误差
    error(data, recdata)
    # 还原降维后的数据
    newImg = Image.fromarray(recdata * 100)
    newImg.show()
    # error(data, recdata)


if __name__ == '__main__':
    data = loadImage("D:/black/pic/suns.jpg")
    pca = PCA(n_components=10).fit(data)
    # 降维
    x_new = pca.transform(data)
    # 还原降维后的数据到原空间
    recdata = pca.inverse_transform(x_new)
    # 计算误差
    error(data, recdata)
    # 还原降维后的数据
    newImg = Image.fromarray(recdata*100)
    newImg.show()
    # error(data, recdata)
