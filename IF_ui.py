import wx
import os
import image_bas
import mainpartcon
import img_cut
import ob_dete
import handwrite.wel
import tkinter as tk
from tkinter import filedialog

def getfile():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="数字图像处理综合程序",
                          pos=(100, 100), size=(700, 500))
        panel = wx.Panel(self)  # 创建画板
        # 创建标题，并设置字体
        title = wx.StaticText(panel, label='数字图像处理综合程序', pos=(270, 20))
        font = wx.Font(16, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(font)
        # 创建文本
        self.bt_confirm = wx.Button(panel, label='图像去噪', pos=(50, 200))
        self.bt_confirm.Bind(wx.EVT_BUTTON, self.image_denoising)
        self.bt_cancel = wx.Button(panel, label='直方图均衡化', pos=(150, 200))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.histogram_equalization)
        self.bt_cancel = wx.Button(panel, label='直线检测', pos=(250, 200))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.edgeDetection_sobel)
        self.bt_cancel = wx.Button(panel, label='物体识别', pos=(350, 200))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.bow_feat)
        #记录点
        self.bt_cancel = wx.Button(panel, label='灰度图', pos=(450, 200))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.togray)
        self.bt_cancel = wx.Button(panel, label='轮廓', pos=(50, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.edge)
        self.bt_cancel = wx.Button(panel, label='特征点提取', pos=(150, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getpoint)
        self.bt_cancel = wx.Button(panel, label='高斯模糊', pos=(250, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getmo)
        #待添加
        self.bt_cancel = wx.Button(panel, label='主成分分析', pos=(350, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getmain)
        self.bt_cancel = wx.Button(panel, label='图像分割', pos=(450, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getcut)
        self.bt_cancel = wx.Button(panel, label='物体检测', pos=(550, 300))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getob)
        self.bt_cancel = wx.Button(panel, label='手写数字识别', pos=(550, 200))
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.getnum)

    def image_denoising(self, event):
        main = "D:/black/data/image_denoising/x64/Debug/image_denoising.exe"
        r_v = os.system(main)
    def histogram_equalization(self, event):
        main = "D:/black/data/histogram_equalization/x64/Debug/histogram_equalization.exe"
        r_v = os.system(main)
    def edgeDetection_sobel(self, event):
        main = "D:/black/data/edgeDetection_sobel/x64/Debug/edgeDetection_sobel.exe"
        r_v = os.system(main)
    def bow_feat(self, event):
        main = "D:/black/data/bow_feat/x64/Debug/bow_feat.exe"
        r_v = os.system(main)
    def togray(self, event):
        image_bas.togray()
    def imchsize(self, event):
        image_bas.imchsize()
    def edge(self, event):
        image_bas.edge()
    def getpoint(self, event):
        image_bas.getpoint()
    def getmo(self, event):
        image_bas.getmo()
    #以下测试
    def getmain(self, event):  # 没有event点击取消会报错
        mainpartcon.gatmain()
    def getcut(self, event):  # 没有event点击取消会报错
        img_cut.walk()
    def getob(self, event):  # 没有event点击取消会报错
        ob_dete.getob()
    def getnum(self, event):  # 没有event点击取消会报错
        handwrite.wel.getnum()











if __name__ == '__main__':
    app = wx.App()  # 初始化应用
    frame = MyFrame(parent=None, id=-1)  # 实例MyFrame类，并传递参数
    frame.Show()  # 显示窗口
    app.MainLoop()  # 调用主循环方法
