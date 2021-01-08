# Style_transfer
A Style_transfer software with GUI
## Environment
```
tensorflow==2.3.0
PyQt5==5.15.2
opencv-python==4.4.0
```
```
CUDA>=10.1
CuDNN>=7.6.4
```
## Run with shell
1. Put the style_img in $PATH_ROOT/data/Style_img
2. Put the content_img in $PATH_ROOT/data/Content_img
3. Modify the $PATH_ROOT/cfgs/config.ini with your images name
4. Enter this in your shell
```
python Convert.py
```

## Run with GUI
```
python start.py
```
PS: If you haven't choose any image before click the start button, system will auto chose the first one in the data files



