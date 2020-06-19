import keras
from keras.datasets import fashion_mnist

(data_train, target_train), (data_test, target_test) = fashion_mnist.load_data()

print('data_train:', data_train.shape)
print(target_train)
print('data_test:', data_test.shape)
print(target_test)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
               
import matplotlib.pyplot as plt
%matplotlib inline
for i in range(0,10):
    class_name = class_names[target_train[i]]
    plt.figure(figsize=(3,3))
    plt.imshow(data_train[i])
    plt.axis('off')
    plt.colorbar()
    plt.title(class_name)
    
    
    
num_classes = 10

target_train = keras.utils.to_categorical(target_train, num_classes)
target_test = keras.utils.to_categorical(target_test, num_classes)

print(target_train[5])



data_train = data_train.astype('float32')
data_test = data_test.astype('float32')

#normalize data
data_train /= 255.0
data_test /= 255.0

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose

data_train=data_train.reshape(data_train.shape[0], *(28,28,1))
data_test=data_test.reshape(data_test.shape[0], *(28,28,1))


### bez dropoutów



convNN_1 = Sequential()
convNN_1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_1.add(MaxPooling2D((2, 2)))
convNN_1.add(Flatten())
convNN_1.add(Dense(100))
convNN_1.add(Activation('relu'))
convNN_1.add(Dense(10))
convNN_1.add(Activation('softmax'))
convNN_1.summary()

convNN_2 = Sequential()
convNN_2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_2.add(MaxPooling2D((2, 2)))
convNN_2.add(Dropout(0.15))
convNN_2.add(Flatten())
convNN_2.add(Dense(100))
convNN_2.add(Activation('relu'))
convNN_2.add(Dense(10))
convNN_2.add(Activation('softmax'))
convNN_2.summary()


convNN_3 = Sequential()
convNN_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_3.add(MaxPooling2D((2, 2)))
# convNN_3.add(Dropout(0.1))
convNN_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# convNN_3.add(Dropout(0.1))
convNN_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# convNN_3.add(Dropout(0.1))
convNN_3.add(Flatten())
convNN_3.add(Dense(100))
convNN_3.add(Activation('relu'))
convNN_3.add(Dense(10))
convNN_3.add(Activation('softmax'))
convNN_3.summary()


convNN_3a = Sequential()
convNN_3a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_3a.add(MaxPooling2D((2, 2)))
convNN_3a.add(Dropout(0.1))
convNN_3a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_3a.add(Dropout(0.1))
convNN_3a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_3a.add(Dropout(0.1))
convNN_3a.add(Flatten())
convNN_3a.add(Dense(100))
convNN_3a.add(Activation('relu'))
convNN_3a.add(Dense(10))
convNN_3a.add(Activation('softmax'))
convNN_3a.summary()


convNN_4 = Sequential()
convNN_4.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_4.add(MaxPooling2D((2, 2)))
convNN_4.add(Dropout(0.1))
convNN_4.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_4.add(Dropout(0.1))
convNN_4.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_4.add(Dropout(0.1))
convNN_4.add(Flatten())
convNN_4.add(Dense(1000))
convNN_4.add(Activation('relu'))
convNN_4.add(Dense(500))
convNN_4.add(Activation('relu'))
convNN_4.add(Dense(100))
convNN_4.add(Activation('relu'))
convNN_4.add(Dense(10))
convNN_4.add(Activation('softmax'))
convNN_4.summary()

convNN_6 = Sequential()
convNN_6.add(SeparableConv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_6.add(MaxPooling2D(pool_size=2))
convNN_6.add(Flatten())
convNN_6.add(Dense(100))
convNN_6.add(Activation('relu'))
convNN_6.add(Dense(10))
convNN_6.add(Activation('softmax'))
convNN_6.summary()


convNN_7 = Sequential()
convNN_7.add(DepthwiseConv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_7.add(Flatten())
convNN_7.add(Dense(100))
convNN_7.add(Activation('relu'))
convNN_7.add(Dense(10))
convNN_7.add(Activation('softmax'))
convNN_7.summary()



convNN_8 = Sequential()
convNN_8.add(Conv2DTranspose(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convNN_8.add(Activation('relu'))
convNN_8.add(MaxPooling2D(pool_size=2))
convNN_8.add(Flatten())
convNN_8.add(Dense(100))
convNN_8.add(Activation('relu'))
convNN_8.add(Dense(10))
convNN_8.add(Activation('softmax'))
convNN_8.summary()


#to plot model as png file
from keras.utils import plot_model
model_name="convNN_1"
plot_model(convNN_1, to_file=model_name+'.png')
#to plot model as png file
from keras.utils import plot_model
model_name="convNN_2"
plot_model(convNN_2, to_file=model_name+'.png')
#to plot model as png file
from keras.utils import plot_model
model_name="convNN_3"
plot_model(convNN_3, to_file=model_name+'.png')
#to plot model as png file
from keras.utils import plot_model
model_name="convNN_3a"
plot_model(convNN_3a, to_file=model_name+'.png')
#to plot model as png file
from keras.utils import plot_model
model_name="convNN_4"
plot_model(convNN_4, to_file=model_name+'.png')
#to plot model as png file
from keras.utils import plot_model
model_name="convNN_6"
plot_model(convNN_6, to_file=model_name+'.png')

#to plot model as png file
from keras.utils import plot_model
model_name="convNN_7"
plot_model(convNN_7, to_file=model_name+'.png')

#to plot model as png file
from keras.utils import plot_model
model_name="convNN_8"
plot_model(convNN_8, to_file=model_name+'.png')

#hyberparameters parameters
batch_size = 500
epochs = 10

# select and initiate optimizer
opt_sgd = keras.optimizers.sgd(lr=0.05)

convNN_1.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd = convNN_1.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_2.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_2 = convNN_2.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_3.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_3 = convNN_3.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_3a.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_3a = convNN_3a.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_4.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_4 = convNN_4.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_6.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_6 = convNN_6.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_7.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_7 = convNN_7.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 

convNN_8.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])

run_hist_sgd_8 = convNN_8.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, target_test), shuffle=True, verbose=1) 


print("---MODEL convNN_1---")

plt.plot(run_hist_sgd.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()

plt.plot(run_hist_sgd.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_2---")

plt.plot(run_hist_sgd_2.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_2.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()

plt.plot(run_hist_sgd_2.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_2.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_3---")

plt.plot(run_hist_sgd_3.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_3.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()

plt.plot(run_hist_sgd_3.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_3.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_3a---")

plt.plot(run_hist_sgd_3a.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_3a.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()

plt.plot(run_hist_sgd_3a.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_3a.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_4---")

plt.plot(run_hist_sgd_4.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_4.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()

plt.plot(run_hist_sgd_4.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_4.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_6---")

plt.plot(run_hist_sgd_6.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_6.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()


plt.plot(run_hist_sgd_6.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_6.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_7---")

plt.plot(run_hist_sgd_7.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_7.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()


plt.plot(run_hist_sgd_7.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_7.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()


print("---MODEL convNN_8---")

plt.plot(run_hist_sgd_8.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_sgd_8.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()
plt.show()


plt.plot(run_hist_sgd_8.history["accuracy"], 'r', marker='.', label="Train Accuracy")
plt.plot(run_hist_sgd_8.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
plt.xlabel("epoch")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()

#model accuracy
scores1 = convNN_1.evaluate(data_test, target_test, verbose=True)
print('Test 1 accuracy:', scores1[1])

scores2 = convNN_2.evaluate(data_test, target_test, verbose=True)
print('Test 2 accuracy:', scores2[1])

scores3 = convNN_3.evaluate(data_test, target_test, verbose=True)
print('Test 3 accuracy:', scores3[1])

scores3a = convNN_3a.evaluate(data_test, target_test, verbose=True)
print('Test 3a accuracy:', scores3a[1])

scores4 = convNN_4.evaluate(data_test, target_test, verbose=True)
print('Test 4 accuracy:', scores4[1])


scores6 = convNN_6.evaluate(data_test, target_test, verbose=True)
print('Test 6 accuracy:', scores6[1])

scores7 = convNN_7.evaluate(data_test, target_test, verbose=True)
print('Test 7 accuracy:', scores7[1])

scores8 = convNN_8.evaluate(data_test, target_test, verbose=True)
print('Test 8 accuracy:', scores8[1])


scores_plot = [0, scores1[1], scores2[1], scores3[1], scores3a[1], scores4[1], scores6[1], scores7[1], scores8[1] ]

plt.plot(scores_plot, "o")
plt.xlabel("model")
plt.ylabel("accuracy")
plt.title("CNN learning with SGD")
plt.legend()
plt.grid()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
import pprint
pp = pprint.PrettyPrinter(indent = 4)

def build_model(optimizer, learning_rate, activation, dropout_rate, initializer, num_unit):


    model = Sequential()
    model.add(Flatten())
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation, input_shape=(28, 28, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])
    return model


batch_size = [500][:1]
epochs = [10][:1]
initializer = ['lecun_uniform'][:1]
learning_rate = [0.1][:1]
dropout_rate = [0, 0.1, 0.2, 0.3][:1]
num_unit = [1000, 3000 ][:1]
activation = ['relu', 'sigmoid'][:1]
optimizer = [SGD][:1]



parameters = dict(batch_size = batch_size,
                  epochs = epochs,
                  dropout_rate = dropout_rate,
                  num_unit = num_unit,
                  initializer = initializer,
                  learning_rate = learning_rate,
                  activation = activation,
                  optimizer = optimizer)



model = KerasClassifier(build_fn=build_model, verbose=1)
models = GridSearchCV(estimator = model, param_grid=parameters, n_jobs=1)


best_model = models.fit(data_train, target_train)
print('Best model :')
pp.pprint(best_model.best_params_)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
import pprint
pp = pprint.PrettyPrinter(indent = 4)

def build_model(optimizer, learning_rate, activation, dropout_rate, initializer, num_unit):


    model = Sequential()
    model.add(Flatten())
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation, input_shape=(28, 28, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(num_unit, kernel_initializer=initializer, activation=activation))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])
    return model


batch_size = [10, 50, 100][:1]
epochs = [15][:1]
initializer = ['lecun_uniform'][:1]
learning_rate = [0.1][:1]
dropout_rate = [0][:1]
num_unit = [500][:1]
activation = ['relu'][:1]
optimizer = [SGD][:1]



parameters = dict(batch_size = batch_size,
                  epochs = epochs,
                  dropout_rate = dropout_rate,
                  num_unit = num_unit,
                  initializer = initializer,
                  learning_rate = learning_rate,
                  activation = activation,
                  optimizer = optimizer)



model = KerasClassifier(build_fn=build_model, verbose=1)
models = GridSearchCV(estimator = model, param_grid=parameters, n_jobs=1)


best_model = models.fit(data_train, target_train)
print('Best model :')
pp.pprint(best_model.best_params_)




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time
import matplotlib.pyplot as plt
#from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
#from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, array_to_img
import os
from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU



(data_train, target_train), (data_test, target_test) = fashion_mnist.load_data()


data_train=np.dstack([data_train] * 3)
data_test=np.dstack([data_test]*3)
data_train.shape
data_test.shape

data_train = data_train.reshape(-1, 28,28,3)
data_test= data_test.reshape (-1,28,28,3)
data_train.shape
data_test.shape


data_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in data_train])
data_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in data_test])
data_train.shape
data_test.shape

data_train = data_train / 255.
data_test = data_test / 255.
data_train = data_train.astype('float32')
data_test = data_test.astype('float32')


train_Y_one_hot = to_categorical(target_train)
test_Y_one_hot = to_categorical(target_test)



train_X,valid_X,train_label,valid_label = train_test_split(data_train, train_Y_one_hot, test_size=0.2, random_state=13)


IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 16

train_X = preprocess_input(train_X)
valid_X = preprocess_input(valid_X)
test_X  = preprocess_input (data_test)

conv_base = VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
                 )
conv_base.summary()

train_features = conv_base.predict(np.array(train_X), batch_size=BATCH_SIZE, verbose=1)
test_features = conv_base.predict(np.array(test_X), batch_size=BATCH_SIZE, verbose=1)
val_features = conv_base.predict(np.array(valid_X), batch_size=BATCH_SIZE, verbose=1)
np.savez("train_features", train_features, train_label)
np.savez("test_features", test_features, target_test)
np.savez("val_features", val_features, valid_label)

train_features_flat = np.reshape(train_features, (48000, 1*1*512))
test_features_flat = np.reshape(test_features, (10000, 1*1*512))
val_features_flat = np.reshape(val_features, (12000, 1*1*512))

NB_TRAIN_SAMPLES = train_features_flat.shape[0]
NB_VALIDATION_SAMPLES = val_features_flat.shape[0]
NB_EPOCHS = 10

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=(1*1*512)))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(),
    metrics=['acc'])


reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=2,
    min_lr=0)

eary_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto')

callbacks = [reduce_learning, eary_stopping]


history_vgg16 = model.fit(
    train_features_flat,
    train_label,
    epochs=NB_EPOCHS,
    validation_data=(val_features_flat, valid_label),
    callbacks=callbacks
)

acc = history_vgg16.history['acc']
val_acc = history_vgg16.history['val_acc']
loss = history_vgg16.history['loss']
val_loss = history_vgg16.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()



from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

(data_train, target_train), (data_test, target_test) = fashion_mnist.load_data()


%matplotlib inline
epochs = 5
batch_size = 100
data_augmentation = False
img_size = 28

num_classes = 10
num_filters = 64
num_blocks = 4
num_sub_blocks = 2
use_max_pool = False


x_train = data_train.reshape(data_train.shape[0],img_size,img_size,1)
x_test = data_test.reshape(data_test.shape[0],img_size,img_size,1)
input_size = (img_size, img_size,1)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255



#Converting labels to one-hot vectors
y_train = keras.utils.to_categorical(target_train, num_classes)
y_test = keras.utils.to_categorical(target_test,num_classes)


inputs = Input(shape=input_size)
x = Conv2D(num_filters, padding='same', 
           kernel_initializer='he_normal', 
           kernel_size=7, strides=2,
           kernel_regularizer=l2(1e-4))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

if use_max_pool:
    x = MaxPooling2D(pool_size=3,padding='same', strides=2)(x)
    num_blocks =3
    

for i in range(num_blocks):
    for j in range(num_sub_blocks):
        strides = 1
        is_first_layer_but_not_first_block = j == 0 and i > 0
        
        
        if is_first_layer_but_not_first_block:
            strides = 2
        #Creating residual mapping using y
        y = Conv2D(num_filters,
                   kernel_size=3,
                   padding='same',
                   strides=strides,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(num_filters,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(y)
        y = BatchNormalization()(y)
        
        
        if is_first_layer_but_not_first_block:
            x = Conv2D(num_filters,
                       kernel_size=1,
                       padding='same',
                       strides=2,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

    num_filters = 2 * num_filters
    
x = AveragePooling2D()(x)
y = Flatten()(x)
outputs = Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()


# checkpoint = ModelCheckpoint(filepath=filepath,
#                              verbose=1,
#                              save_best_only=True)
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)
# callbacks = [checkpoint, lr_reducer]


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    
    
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
