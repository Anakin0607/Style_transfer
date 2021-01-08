import tensorflow as tf
from tools import Load_img
import os
from cfgs.config import *
from tools import Tensor2Image
import cv2 as cv
import time
from tqdm import tqdm

gpu = tf.config.experimental.list_physical_devices(device_type='GPU') # apply memory as use
for k in range(len(gpu)):
    tf.config.experimental.set_memory_growth(gpu[k], True)

config_data.read(config_file)
content_img = Load_img(img_path)
style_img = Load_img(style_path)

x = tf.keras.applications.vgg19.preprocess_input(content_img*255)
x = tf.image.resize(x,(224,224))

vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')

content_layers = ['block5_conv2'] # get the feature map

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1'] 
                #'block5_conv1'] # get the style layers

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names): # create vgg models which returns a list of intermediate output values
    vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input],outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_img*255)

#print the information of the style_layers
print("Information of style layers:")

for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

def Gram_matrix(input_tensor): # get the style matrix by the feature map
    result = tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2],tf.float32)
    return result/num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self,style_layers,content_layers):
        super(StyleContentModel,self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self,inputs):
        #Input is expected in float [0,1]
        inputs = inputs * 255.0
        preprocess_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocess_input)
        style_outputs,content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])

        style_outputs = [Gram_matrix(style_output) 
                            for style_output in style_outputs]
        
        content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers,content_layers)

results = extractor(tf.constant(content_img))

style_results = results['style'] # get the gram matrix of the style_layers

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

style_targets = extractor(style_img)['style']
content_targets = extractor(content_img)['content']

img = tf.Variable(content_img) # contain the image to optimize

def clip_0_1(img):#normalization
    return tf.clip_by_value(img,clip_value_min = 0.0,clip_value_max = 1.0)

optimizer = tf.optimizers.Adam(learning_rate=0.02,beta_1=0.99,epsilon=1e-1)

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) # use the mean square loss
                           for name in style_outputs.keys()]) 
    
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                            for name in content_outputs.keys()])

    content_loss *= content_weight / num_content_layers
    loss = content_loss + style_loss

    return loss

def High_pass(img):
    x_var = img[:,:,1:,:] - img[:,:,:-1,:]
    y_var = img[:,1:,:,:] - img[:,:-1,:,:]

    return x_var,y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

#x_deltas, y_deltas = high_pass_x_y(content_image)
#
#plt.figure(figsize=(14,10))
#plt.subplot(2,2,1)
#imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")
#
#plt.subplot(2,2,2)
#imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")
#
#x_deltas, y_deltas = high_pass_x_y(image)
#
#plt.subplot(2,2,3)
#imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")
#
#plt.subplot(2,2,4)
#imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

#plt.figure(figsize=(14,10))
#
#sobel = tf.image.sobel_edges(content_image)
#plt.subplot(1,2,1)
#imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
#plt.subplot(1,2,2)
#imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")

@tf.function()
def train_step(image):
    with tf.GradientTape() as Tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)
        
    grad = Tape.gradient(loss,image)
    optimizer.apply_gradients([(grad,image)])
    image.assign(clip_0_1(image))

def train_start():
    #strat = time.time()
    #
    #with tqdm(total = epochs*step_per_epoch) as bar:
    #    for n in range(epochs):
    #        for m in range(step_per_epoch):
    train_step(img)
    #            bar.update(1)
    #
    #end = time.time()
    #total_time = end-strat
    #
    result_image = Tensor2Image(img)

    return result_image#,total_time

if __name__ == "__main__":
    time,img = train_start()
    print("Total time is%.2f"%(time))
    cv.imshow("converted",img)
    cv.waitKey()