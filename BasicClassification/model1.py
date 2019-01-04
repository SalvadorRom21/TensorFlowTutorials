#Salvador Romero (TheHeartlessOne)
#This file will contain different models to help classify items of cloathing 
#from the fashion MNIST dataset 
#Different models will be made with the keras API and we will explore the one that will end 
#up with the highest accuracy while having a low loss cost. 

#This imports tesorflow and keras API
import tensorflow as tf
from tensorflow import keras
#This imports the helper libraries numpy and matplotlib.pyplot
import numpy as np 
import matplotlib.pyplot as plt 

#This will import the Fashion MNIST dataset 
#The data set contains 70,000 images of clothing from 10 different categories
#The data set conatains 60,000 images of clothing that will be used to train the models 
#The data set also contains 10,000 images of clothing that will be used to evaluate how accurate
#the models is when presented with new data.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#These are the ten categories that the data set contains.
class_names = ['T-shirt/top','Trouser','Pullover','Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']

#Each image in the training data set will be using images that are 28x28 pixels
#This deals with the 60,000 images that we will need to train the models
train_images.shape
len(train_labels)
train_labels
#The lines below deal with the 10,000 images that we will need to evaluate the accuracy of the model
test_images.shape
len(test_labels)

#We need to process the data before training the model.
#Each pixel in an image contains a value between 0 and 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#To make it easier for the model to handle the data we need to normalize it so that the values
#scale the values from 0-255 to 0-1

train_images = train_images / 255.0
test_images = test_images / 255.0 

plt.figure(figsize=(10,10))

for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])

#This will make the model that our data will be running through
#The first layer will have 748 imputs representing the 28x28 images
#The second layer will have 128 fully connected nodes
#The third layer will be the output layer. We have 10 nodes here
#one for each category in our data set.
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(748, activation = tf.nn.relu),
  keras.layers.Dense(374, activation = tf.nn.relu),
  keras.layers.Dense(187, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
	])

#This adds more settign to the model.
#The loss fucntion messures how accurate the model is doing during training
#The optimizer is how the model is updated based on the data and the loss function
#The metrics are used to monitor the training and testing steps
model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

#This is the actual training of the model
#Epochs is the number of time the training model uses the training data set to train the model
model.fit(train_images, train_labels, epochs = 5)

#Evaluating the accuracy of the model.

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

#This line of code determines the predictions of some images.

predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0])

test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')






#This print out the current tensorflow version.
print(tf.__version__)