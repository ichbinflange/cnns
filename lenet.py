from copyreg import pickle
from gc import callbacks
from statistics import mode
from tkinter.tix import MAX
from cv2 import imread
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import os
import glob
from PIL import Image,ImageOps
import cv2
import matplotlib.pyplot as plt

###dataaugumentation
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip,RandomRotation

def DisplayImageToArray(img,size):

    #image = Image.open(img)
    image = ImageOps.fit(img,size,Image.ANTIALIAS)
    img_frompil = np.array(image)

    return img_frompil

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = glob.glob(os.path.join(os.getcwd(),r'cnns/cifar-10-batches-py',"*"))


trainbatch = []
testbatch = []
for file in files:
    if 'data' in file:
        trainbatch.append(unpickle(file))
    elif 'test' in file:
        testbatch.append(unpickle(file))


#taking only one batch for training
traindata = trainbatch[1]

ximages = traindata[b'data']
ytrain = traindata.get(b'labels')


xtrain = [img.reshape(3,32,32).transpose([1, 2, 0]) for img in ximages]

xtrain = np.asarray(xtrain,np.float32)

ytrain = to_categorical(ytrain,10)

xtrain = xtrain/255.0


##create the model

model = tf.keras.Sequential()
###add some augumentation
model.add(RandomFlip('horizontal'))
model.add(RandomFlip('vertical'))
model.add(RandomRotation(0.2))
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (32,32,3),padding ='valid'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu',input_shape = (32,32,3),padding ='valid'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu',input_shape = (32,32,3),padding ='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add((Flatten()))
model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))




es= EarlyStopping(monitor= 'val_loss',mode= 'auto',patience=4)

model.compile(optimizer ='adam',loss =tf.keras.losses.BinaryCrossentropy(),metrics = ['accuracy'])
model.build(input_shape = xtrain.shape)

#model.summary()
history = model.fit(xtrain,ytrain,validation_split = 0.2,epochs = 50,batch_size = 32,callbacks =[es] )

##plot accuracy
plt.plot(history.history ['accuracy'])
plt.plot(history.history ['val_accuracy'])
plt.title('model.accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

##plot loss
plt.plot(history.history ['loss'])
plt.plot(history.history ['val_loss'])
plt.title('model.loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()