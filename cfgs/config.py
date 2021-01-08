import os
from configparser import ConfigParser

config_file = os.path.join(os.path.join(os.getcwd(),"cfgs"),"config.ini")
config_data = ConfigParser()
config_data.read(config_file)


img_path = config_data.get('path','original')
style_path = config_data.get('path','style')

style_weight = 1e-2 # give two weighted losses in the total loss
content_weight = 1e5

epochs = 10
step_per_epoch = 100 # don't set the bigger parameter than epoch*step_per_epoch=1000

total_variation_weight = 20