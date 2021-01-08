import sys
import os

from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QGraphicsPixmapItem,QGraphicsScene
from PyQt5 import QtGui
from PyQt5.QtCore import QThread,pyqtSignal
from gui.ui import Ui_Form

import time
import cv2 as cv
from configparser import ConfigParser

config_file = os.path.join(os.path.join(os.getcwd(),"cfgs"),"config.ini")
config_data = ConfigParser()
config_data.read(config_file)

class MyMainForm(QMainWindow,Ui_Form):
    def __init__(self,parent=None):  

        super(MyMainForm,self).__init__(parent)

        self.Ui = Ui_Form()
        self.Ui.setupUi(self)

        #Activate the convert thread
        self.Work = ConvertThread()

        self.Ui.File_choose_Button.clicked.connect(self.Choose_Ori_img) # choose the origianl img by the choose button
        self.Ui.File_choose_Button_2.clicked.connect(self.Choose_Style_img) # choose the style img by the choose button
        self.Ui.Start_Button.clicked.connect(self.Start_train)
        self.Ui.Save_Button.clicked.connect(self.SaveImage)

       
    def Choose_Img(self):
        img_name,img_type = QFileDialog.getOpenFileName(self,"Choose files"
                                                            ,""
                                                            ,"*.jpg;;*.png;;*.All Files(*)") # return two values: file name and file type
        return img_name

    def Show_Img(self,img,Viewer):
        #img_name = self.Choose_Img()
        #self.Ui.File_Edit.setText(img_name)

        #config_data.set('path','original',img_name)# write the path to config 
        with open(config_file,'w') as config:
            config_data.write(config)
            config.close()

        #img = cv.imread(img_name)
        img_RGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img_resized = cv.resize(img_RGB,(Viewer.height(),Viewer.width()))

        Show_image = QtGui.QImage(img_resized,
                                img_resized.shape[1],
                                img_resized.shape[0],
                                img_resized.shape[1]*img_resized.shape[2], # Linewidth four-byte alignment, it must be done if the width can't be divided by 4
                                QtGui.QImage.Format_RGB888)# convert the img format to the qt format
        pix = QtGui.QPixmap.fromImage(Show_image)
        item = QGraphicsPixmapItem(pix) # create pix metas

        scene = QGraphicsScene()
        scene.addItem(item)
        Viewer.setScene(scene)

    def Choose_Ori_img(self):
        img_name = self.Choose_Img()
        
        if not img_name == "": # Prevert user close the chose window without chose any image
            self.Ui.File_Edit.setText(img_name)
            config_data.set('path','original',img_name)# write the path to config
        
            img = cv.imread(img_name)
            self.Show_Img(img,self.Ui.Ori_image_View)
    
    def Choose_Style_img(self):
        img_name = self.Choose_Img()

        if not img_name == "": # Prevert user close the chose window without chose any image
            self.Ui.File_Edit_style.setText(img_name)
            config_data.set('path','style',img_name)# write the path to config 

            img = cv.imread(img_name)
            self.Show_Img(img,self.Ui.Styled_image_View)

    #def Start_train(self):
    #    self.Convert_init()
    #    from Convert import train_start # Load the core model
    #
    #    #for i in range(epochs):
    #        #for j in range(step_per_epoch):
    #    img,time = train_start()
    #            #self.Ui.progressBar.setProperty("value", ((i*step_per_epoch+(j+1))/epochs*step_per_epoch)*100)
    #            #print(i*step_per_epoch+(j+1))
    #
    #    print("Total time: %.2f s"%(time))
    #    self.Show_Img(img,self.Ui.Converted_image_View)

    def Start_train(self):
        self.Convert_init()
        self.Work.start()
        self.Work.trigger.connect(self.PrograssBar_update)
        self.Work.img_trigger.connect(self.Converted_view)
        #img_path = os.getcwd() + "\\Transferd_img\\transferd.jpg"
        #if os.path.exists(img_path):
        #    img = cv.imread()
        #    self.Show_Img(img,self.Ui.Converted_image_View)

    def Converted_view(self,object):
        self.Converted_img = object
        self.Show_Img(object,self.Ui.Converted_image_View)

    def PrograssBar_update(self,str):
        #print(int(str))
        self.Ui.progressBar.setProperty("value",int(str))

    def SaveImage(self):
        img_path = QFileDialog.getSaveFileName(self,"getSaveFileName",
                                                    "./",
                                                    "All files(*);;(*.jpg);;(*.png)")
        #print(img_path[0])
        cv.imwrite(img_path[0],self.Converted_img)
        #cv.imshow("win",self.Converted_img)
        #print(self.Converted_img)

    def Convert_init(self):
        Content_img = config_data.get('path','original')
        style_img = config_data.get('path','style')

        if not os.path.exists(Content_img): # Confirm if the config.ini is currect
            Content_path = os.getcwd() + "\\data\\Content_img"

            if not os.path.exists(Content_path):
                os.mkdir(Content_path)

            Content_img = os.path.join(Content_path,os.listdir(Content_path)[0]) # set the default image if user want to start convert before choose image
            
            config_data.set('path','original',Content_img)
            
        if not os.path.exists(style_img):
            style_path = os.getcwd() + "\\data\\Style_img"

            if not os.path.exists(Content_path):
                os.mkdir(Content_path)

            style_img = os.path.join(style_path,os.listdir(style_path)[0])

            config_data.set('path','style',style_img)

        content_show_img = cv.imread(Content_img)
        self.Show_Img(content_show_img,self.Ui.Ori_image_View)

        style_show_img = cv.imread(style_img)
        self.Show_Img(style_show_img,self.Ui.Styled_image_View)

class ConvertThread(QThread):

    trigger = pyqtSignal(str) #emit the time
    img_trigger = pyqtSignal(object) #emit the img

    def __init__(self):
        super(ConvertThread,self).__init__()

    def run(self):

        from Convert import train_start # Load the core model
        from cfgs.config import epochs,step_per_epoch

        start = time.time()

        for i in range(epochs):
            for j in range(step_per_epoch):
                img = train_start()
                #MyMainForm.Ui.progressBar.setProperty("value", ((i*step_per_epoch+(j+1))/epochs*step_per_epoch)*100)

                now = int((i*step_per_epoch+(j+1))/(epochs*step_per_epoch)*100) # how many percentage of the step

                self.img_trigger.emit(img)
                self.trigger.emit(str(now))
                
        end = time.time()

        total_time = end-start
        print("Total time: %.2f s"%(total_time))  

if __name__ =="__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())