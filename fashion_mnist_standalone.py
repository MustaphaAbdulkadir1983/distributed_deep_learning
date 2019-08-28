
from __future__ import print_function
import keras
from keras.layers.normalization import BatchNormalization
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
from mpi4py import MPI
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from numpy import genfromtxt


batch_size = 128
epochs = 1
nb_epochs = 24
num_classes = 10
num_workers = 4
# input image dimensions
img_rows, img_cols = 28, 28

model_loss = []
model_accuracy = []
validation_loss = []
validation_accuracy = []

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

sub_data_size = len(x_train)//num_workers

# function for building and compiling the model
def build_model(num_classes):
	print("Building the model")
	print("==================")
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
	return model

# function for doing one epoch	
def do_one_epoch(model, train_data, train_label):
	result=model.fit(train_data, train_label,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_split=0.1)
	model_loss.append(result.history['loss']) #Now append the loss after the training to the list.
	model_accuracy.append(result.history['acc'])
	validation_loss.append(result.history['val_loss'])
	validation_accuracy.append(result.history['val_acc'])
	return model

# function for evaluating the model as well as visualizing
def evaluate_model(model, test_data, test_label):
	score = model.evaluate(test_data, test_label, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

# function for finding the average of the weights	
def average_weights(all_weights):
	new_weights = []
	for weights_list_tuple in zip(*all_weights):
		new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
	return new_weights

# main method
if __name__ == '__main__':

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	# check to ensure that only 4 workers are specipied in the command line
	if size!=4 and rank==0:
		print(rank, "Data are prepared only for 4 workers")
	if size!=4:
		sys.exit(-1)

	# declaring neighbors of each worker
	if rank == 0:
		neighbors = [1,2,3]
	if rank == 1:
		neighbors = [0,2,3]
	if rank == 2:
		neighbors = [0,1,3]
	if rank == 3:
		neighbors = [0,1,2]

	'''All the 4 workers build, save, train and evaluate the model independently,
	they only exchange all thier weights at end of each epoch'''
	for i in range(0, size):
		if rank == i:
			print("I am worker with number: ", i, "I will will build and the model")
			# building the model
			model = build_model(num_classes) 
			# saving the model and its weights
			model.save("model_"+str(i)+".h5")
			print("Model built and saved")
			
			# training the model
			for epoch in range(nb_epochs):
				print("worker: ", rank, "epoch: ", epoch, "of:", nb_epochs)
				train_model=do_one_epoch(model, x_train[i*sub_data_size:(i+1)*sub_data_size], 
					y_train[i*sub_data_size:(i+1)*sub_data_size])
						
				np.savetxt("standalone_model_loss"+str(i)+".csv", model_loss, delimiter=",")
				np.savetxt("standalone_model_accuracy"+str(i)+".csv", model_accuracy, delimiter=",")
				np.savetxt("standalone_validation_loss"+str(i)+".csv", validation_loss, delimiter=",")
				np.savetxt("standalone_validation_accuracy"+str(i)+".csv", validation_accuracy, 	delimiter=",")

			# evaluating the model
			print("worker: ", i, "result is: ")
			evaluate_model(train_model, x_test, y_test)
			
		







			
			
		

