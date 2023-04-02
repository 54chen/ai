from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
import os
from tensorflow.keras import callbacks
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

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
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)
print("Total disease classes are: {}".format(len(diseases)))


train_datagen_aug = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   fill_mode="nearest",
                                   rotation_range = 20,
                                   width_shift_range=0.2,
                                    height_shift_range=0.2,
                                   horizontal_flip = True)

test_datagen_aug = ImageDataGenerator( rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   horizontal_flip = True)


training_set_aug = train_datagen_aug.flow_from_directory(directory= train_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=16,
                                               class_mode='categorical') # for 2 class binary 
label_map = (training_set_aug.class_indices)
print("Target Classes Mapping Dict:\n")
print(label_map)

test_set_aug = test_datagen_aug.flow_from_directory(directory= valid_dir,
                                               target_size=(224, 224), # As we choose 64*64 for our convolution model
                                               batch_size=16,
                                               class_mode='categorical',
                                               shuffle=False) # for 2 class binary

with strategy.scope():
    image_input = Input(shape=(224,224,3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
    num_classes = 38

    x = base_model(image_input, training = False)
    base_model.trainable = False
#     x = Dense(256,activation = "relu")(x)
#     x = Dropout(0.2)(x)

#     x = Dense(128,activation = "relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)


    # 构建新模型
    model = Model(inputs=image_input, outputs=output)

    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



with strategy.scope():
    early_stopping_cb = callbacks.EarlyStopping(monitor="loss", patience=10)
    history = model.fit(training_set_aug,
                        epochs=30,
                        verbose=1,
                        steps_per_epoch=150,
                        callbacks=[early_stopping_cb]
                       )

model.evaluate(test_set_aug)

import matplotlib.pyplot as plt

# Plotting
hist = history.history

# Plot accuracy and loss
plt.plot(hist["accuracy"], label="accuracy")
plt.plot(hist["loss"], label="loss")

if "val_accuracy" in hist and "val_loss" in hist:
    plt.plot(hist["val_accuracy"], label="val_accuracy")
    plt.plot(hist["val_loss"], label="val_loss")

# Add the labels and legend
plt.ylabel("Accuracy / Loss")
plt.xlabel("Epochs #")
plt.legend()

# Finally show the plot
plt.show()