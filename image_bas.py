import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image
from pylab import *
from matplotlib.font_manager import FontProperties
from scipy.ndimage import filters

def getfile():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


# 添加中文字体支持
def togray():
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
    figure()
    filename=getfile()
    pil_im = Image.open(filename)
    gray()
    subplot(121)
    title(u'原图',fontproperties=font)
    axis('off')
    imshow(pil_im)
    pil_im = Image.open(filename).convert('L')
    subplot(122)
    title(u'灰度图',fontproperties=font)
    axis('off')
    imshow(pil_im)
    show()


def imchsize():
    filename = getfile()
    pil_im = Image.open(filename)
    out = pil_im.resize((128, 128))
    imshow(pil_im)
    show()


def edge():
    filename = getfile()
    # 读取图像到数组中
    im = array(Image.open(filename).convert('L'))
    # 新建一个图像
    figure()
    # 不使用颜色信息
    gray()
    # 在原点的左上角显示轮廓图像
    contour(im, origin='image')
    axis('equal')
    axis('off')
    figure()
    hist(im.flatten(), 128)
    show()


def getmo():
    filename = getfile()
    im = array(Image.open(filename))
    im2 = zeros(im.shape)
    for i in range(3):
        im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
    im2 = uint8(im2)
    imshow(im2)
    show()

def getpoint():
    img = cv2.imread(getfile())
    img = cv2.resize(img, (136 * 3, 76 * 3))
    cv2.imshow("original", img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray, None)

    cv2.drawKeypoints(image=img,
                      outImage=img,
                      keypoints=keypoints,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(51, 163, 236))
    cv2.imshow("SIFT", img)

    # 使用SURF
    img = cv2.resize(img, (136 * 3, 76 * 3))

    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptor = surf.detectAndCompute(gray, None)

    cv2.drawKeypoints(image=img,
                      outImage=img,
                      keypoints=keypoints,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(51, 163, 236))
    cv2.imshow("SURF", img)

    img = cv2.resize(img, (136 * 3, 76 * 3))

    cv2.waitKey(0)
    cv2.destroyAllWindows()