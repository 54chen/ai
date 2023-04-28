import os

import tensorflow as tf
from tensorflow.keras import Model, callbacks
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Convolution2D,Activation,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)

data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
test_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)
print("Total disease classes are: {}".format(len(diseases)))

train_datagen_aug = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   fill_mode="nearest",
                                   rotation_range = 20,
                                   width_shift_range=0.2,
                                    height_shift_range=0.2,
                                   horizontal_flip = True,
                                   validation_split=0.2) # set validation split


test_datagen_aug = ImageDataGenerator( rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   horizontal_flip = True)


training_set_aug = train_datagen_aug.flow_from_directory(directory= train_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=128,
                                               class_mode='categorical',
                                               subset='training')


validation_set_aug = train_datagen_aug.flow_from_directory(directory= train_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=128,
                                               class_mode='categorical',
                                               subset='validation',
                                               shuffle=False)

label_map = (training_set_aug.class_indices)
print("Target Classes Mapping Dict:\n")
print(label_map)
label_map = (validation_set_aug.class_indices)
print("Target Classes Mapping Dict:\n")
print(label_map)


test_set_aug = test_datagen_aug.flow_from_directory(directory= test_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=128,
                                               class_mode='categorical') # for 2 class binary
label_map = (test_set_aug.class_indices)
print("Target Classes Mapping Dict:\n")
print(label_map)

with strategy.scope():
    # Instantiate an empty sequential model
    model = Sequential(name="Alexnet")
    # 1st layer (conv + pool + batchnorm)
    model.add(Conv2D(filters= 96, kernel_size= (11,11), strides=(4,4), padding='valid', kernel_regularizer=l2(0.0005), input_shape = (224,224,3)))
    model.add(Activation('relu'))  #<---- activation function can be added on its own layer or within the Conv2D function
    model.add(MaxPool2D(pool_size=(3,3), strides= (2,2), padding='valid'))
    model.add(BatchNormalization())
        
    # 2nd layer (conv + pool + batchnorm)
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
                
    # layer 3 (conv + batchnorm)      <--- note that the authors did not add a POOL layer here
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
            
    # layer 4 (conv + batchnorm)      <--- similar to layer 3
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
                
    # layer 5 (conv + batchnorm)  
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(Flatten())

    # layer 6 (Dense layer + dropout)  
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.5))

    # layer 7 (Dense layers) 
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.5))
                              
    # layer 8 (softmax output layer) 
    model.add(Dense(units = 38, activation = 'softmax'))

    print(model.summary())
    
    optimizer = SGD(lr = 0.01, momentum = 0.9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy',TopKCategoricalAccuracy(k=1, name="top1")])

early_stopping_cb = callbacks.EarlyStopping(monitor="val_loss", patience=3)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                              factor=0.2, 
                                              patience=2,
                                              verbose=1, 
                                              min_lr=1e-7)
checkpoint_path = "fine_tune_checkpoints/"
model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      monitor="val_loss")

history = model.fit(training_set_aug,
                    epochs=10,
                    verbose=1,
                    callbacks=[early_stopping_cb, model_checkpoint, reduce_lr],
                    validation_data = validation_set_aug, 
                    # steps_per_epoch = 20,
                    # validation_steps = 20
                    )

model.load_weights(checkpoint_path)
model.evaluate(test_set_aug)



# Plotting
hist = history.history
def show_plt(type):
    if type == 1:
        plt.plot(hist["accuracy"], label="accuracy")
        plt.plot(hist["val_accuracy"], label="val_accuracy")
        plt.ylabel("Aaccuracy")
        plt.xlabel("Epochs #")
        plt.legend()
        plt.show()
    else:
        plt.plot(hist["loss"], label="loss")
        plt.plot(hist["val_loss"], label="val_loss")
        plt.ylabel("Losss")
        plt.xlabel("Epochs #")
        plt.legend()
        plt.show()
        
show_plt(1)
show_plt(0)