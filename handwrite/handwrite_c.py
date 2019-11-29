'''
    功能：
        利用训练好的模型，进行实时手写体识别
    作者：yuhansgg
    博客: https://blog.csdn.net/u011389706
    日期: 2018/08/06
'''
import tensorflow as tf
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image


class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.resize(284, 330)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点

        # 添加一系列控件
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('识别结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 380, 380)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素

        recognize_result = self.recognize_img(im)  # 调用识别函数

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):  # 手写体识别函数
        myimage = img.convert('L')  # 转换成灰度图
        tv = list(myimage.getdata())  # 获取图片像素值
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑

        init = tf.global_variables_initializer()
        saver = tf.train.Saver  

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.import_meta_graph('D:/black/py_data/handwrite/MI/minst_cnn_model.ckpt.meta')  # 载入模型结构
            saver.restore(sess, 'D:/black/py_data/handwrite/MI/minst_cnn_model.ckpt')  # 载入模型参数

            graph = tf.get_default_graph()  # 加载计算图
            x = graph.get_tensor_by_name("x:0")  # 从模型中读取占位符变量
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            y_conv = graph.get_tensor_by_name("y_conv:0")  # 关键的一句  从模型中读取占位符变量

            prediction = tf.argmax(y_conv, 1)
            predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
            print(predint[0])
        return predint[0]