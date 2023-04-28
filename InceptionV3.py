import os

import tensorflow as tf
from tensorflow.keras import Model, callbacks
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2,
                                                        preprocess_input)
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


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
                                               batch_size=64,
                                               class_mode='categorical',
                                               subset='training')


validation_set_aug = train_datagen_aug.flow_from_directory(directory= train_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=64,
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
                                               batch_size=64,
                                               class_mode='categorical') # for 2 class binary
label_map = (test_set_aug.class_indices)
print("Target Classes Mapping Dict:\n")
print(label_map)

with strategy.scope():
    image_input = Input(shape=(224,224,3))
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
#     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    num_classes = 38

    x = base_model(image_input, training = False)
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False
#     x = Dense(256,activation = "relu")(x)
#     x = Dropout(0.2)(x)

#     x = Dense(128,activation = "relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
#     x = Dense(256,activation = "relu")(x)

    output = Dense(num_classes, activation='softmax')(x)


    # 
    model = Model(inputs=image_input, outputs=output)

    print(model.summary())
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy',TopKCategoricalAccuracy(k=1, name="top1")])

    early_stopping_cb = callbacks.EarlyStopping(monitor="val_loss", patience=3)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                              factor=0.2, 
                                              patience=2,
                                              verbose=1, 
                                              min_lr=1e-7)
history = model.fit(training_set_aug,
                    epochs=30,
                    verbose=1,
                    callbacks=[early_stopping_cb, reduce_lr],
                    validation_data = validation_set_aug, 
                    )

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