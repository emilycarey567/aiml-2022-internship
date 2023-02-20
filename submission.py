#Predicting a Brain tumour in a MRI 
#Project by Emily Carey

#This script loads an image dataset from a directory, trains a convolutional neural network (CNN) 
#to classify images in the dataset as either containing a tumour or not, and evaluates the performance 
#of the trained model on a test set. It then demonstrates how the trained model can be used to predict 
#whether a new image contains a tumour.

# Import necessary libraries
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import imghdr
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.setMemoryGrowth(gpu, True)

# Remove any dodgy images with incorrect extensions
tf.config.list_physical_devices('GPU')
dataDir = 'data' 
imageExts = ['jpeg','jpg', 'bmp', 'png']
for imageClass in os.listdir(dataDir): 
    for image in os.listdir(os.path.join(dataDir, imageClass)):
        imagePath = os.path.join(dataDir, imageClass, image)
        try: 
            img = cv2.imread(imagePath)
            tip = imghdr.what(imagePath)
            if tip not in imageExts: 
                print('Image not in ext list {}'.format(imagePath))
                os.remove(imagePath)
        except Exception as e: 
            print('Issue with image {}'.format(imagePath))

# os.remove(image_path)
# Load image data from directory
data = tf.keras.utils.image_dataset_from_directory('data')

# Get iterator for data batches
dataIterator = data.as_numpy_iterator()

# Get a batch of data for visualization
batch = dataIterator.next()

# Plot the first four images in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Normalize pixel values between 0 and 1
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# Split data into train, validation, and test sets
trainSize = int(len(data)*.7)
valSize = int(len(data)*.2)
testSize = int(len(data)*.1)
trainSize
train = data.take(trainSize)
val = data.skip(trainSize).take(valSize)
test = data.skip(trainSize+valSize).take(testSize)
train

# Define a sequential model with convolutional and dense layers
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and accuracy metric
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Define a tensorboard callback to log training information
logdir='logs'
tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model and log training information to tensorboard
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboardCallback])

# Plot loss and accuracy graphs for the training process
# The hist object contains the history of the training 
#process, and the graphs plot the training and validation loss/accuracy over epochs. 
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Training and Validation Loss Over Epochs', fontsize=15)
plt.legend(loc="upper left")
plt.show()
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Training and Validation Accuracy Over Epochs', fontsize=15)
plt.legend(loc="upper left")
plt.show()

#Calculates and prints the precision, recall, and binary accuracy metrics of the model on a test dataset
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.updateState(y, yhat)
    re.updateState(y, yhat)
    acc.updateState(y, yhat)
print(pre.result(), re.result(), acc.result())

#Loads an image of a tumour using OpenCV, displays the original image, resizes the image to (256, 256), 
#displays the resized image, and then predicts whether the resized image contains a tumour or not. 
img = cv2.imread('tumour.jpg')
plt.imshow(img)
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = model.predict(np.expand_dims(resize/255, 0))
yhat
if yhat > 0.5: 
    print(f'Predicted Tumour')
else:
    print(f'Predicted no Tumour')