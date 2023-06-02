import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Reshape, Activation, Conv2D,UpSampling2D,Flatten,Dense,Dropout,BatchNormalization,ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import seaborn as sns
from IPython import display


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(10)
# The dimension of our random noise vector.
random_dim = 100
batch_size = 256
# Create batches and shuffle the dataset
CNN = False

try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)

def load_minst_data():
    # load the data
    
    train_data = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
    test_data = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

    train_data = pd.concat([train_data, test_data])

    print("Train Shape --> ",train_data.shape)

    # # how many of which numbers are there?
    # plt.figure(figsize=(15,7))
    # sns.countplot(x=train_data["label"], palette="cubehelix")
    # plt.title("Number of digit classes")
    # print(" \t # Train Data value counts # \n",train_data["label"].value_counts())
    # plt.show()

    X_train = train_data.drop(["label"], axis = 1)
    y_train = train_data.label
    
    X_train = (X_train.astype(np.float32)-127.5)/127.5
    y_train = y_train.astype(np.float32)/10
    X_train = pd.concat([X_train, y_train], axis=1)
    X_train = X_train.rename(columns={"y_train": "label"})

    X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(X_train.shape[0]).batch(batch_size)

    return X_train

def get_CNN_generator():
    model = Sequential(name="cnn-g")
    
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=random_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.add(Flatten())
    model.add(Dense(784))
    model.summary()
    # model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model


def get_CNN_discriminator():
    model = Sequential(name="cnn-d")
    model.add(Dense(784, input_shape=(784,)))
    model.add(Reshape((28,28,1)))

    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0002, 0.5))
    return model

def get_generator():
    generator = Sequential(name="generator")
    generator.add(Dense(256, input_dim=random_dim + 1))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784, activation='tanh'))
    # generator.compile(loss='binary_crossentropy', optimizer='rmsprop')

    generator.summary()
    return generator

def get_discriminator():
    
    input1 = Input(shape=(784,))
    # label
    input2 = Input(shape=(1,))
 
    merged = concatenate([input1, input2])

    i1 = Dense(1024, activation=LeakyReLU(0.2))(merged)    
 
    hidden1 = Dense(1024)(i1)
    hidden1 = LeakyReLU(0.2)(hidden1)
    hidden1 = Dropout(0.3)(hidden1)

    hidden2 = Dense(512)(hidden1)
    hidden2 = LeakyReLU(0.2)(hidden2)
    hidden2 = Dropout(0.3)(hidden2)

    hidden3 = Dense(256)(hidden2)
    hidden3 = LeakyReLU(0.2)(hidden3)
    hidden3 = Dropout(0.3)(hidden3)

    # 定义输出层
    output_layer = Dense(1, activation='sigmoid')(hidden3)

    # 构建模型
    discriminator = Model(inputs=[input1, input2], outputs=output_layer, name='discriminator')

    # 编译模型
    discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')


    discriminator.summary()
    return discriminator

def get_gan_network(discriminator, random_dim, generator):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    z = Input(shape=(random_dim + 1,))
    label = Input(shape=(1,))

    # the output of the generator (an image)
    x = generator(z)

    gan_output = discriminator([x,label])
    gan = Model(inputs=[z,label], outputs=gan_output)     
    gan.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0002, 0.5))
    return gan

dis_accuracy = []
dis_loss = []
gan_accuracy = []
gan_loss = []

def train(epochs=1):
    # Get the training and testing data
    x_train = load_minst_data()

    # Build our GAN netowrk
    with strategy.scope():
        generator = get_generator()
        discriminator = get_discriminator()
        if CNN:
            generator = get_CNN_generator()
            discriminator = get_CNN_discriminator() 

        gan = get_gan_network(discriminator, random_dim, generator)

    pbar = tqdm(total=epochs,mininterval=1,ncols=100)

    for e in range(1, epochs+1):
        print("\n",'-'*15, 'Epoch %d' % e, '-'*15, "\n")
        d_l, d_a, g_l, g_a = 0,0,0,0
        for image_batch in x_train:
            b_size = image_batch.shape[0]
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[b_size, random_dim])
            fake_labels = np.random.choice(np.arange(0.0, 1.0, 0.1), size=(b_size, 1))
            noise = tf.concat([noise, fake_labels], axis=1)
            # Generate fake MNIST images
            generated_images = generator.predict(noise,verbose=0)
            real_images = image_batch[:, :-1]
            real_lables = image_batch[:, -1]
            real_lables = tf.reshape(real_lables, [b_size, 1])
            X = tf.concat([real_images, generated_images], axis=0)
            L = tf.concat([real_lables, fake_labels], axis=0)

            # Labels for generated and real data
            y_dis = np.zeros(2 * image_batch.shape[0])
            # One-sided label smoothing
            y_dis[:image_batch.shape[0]] = 1

            # Train discriminator
            discriminator.trainable = True
            dis_hist = discriminator.train_on_batch([X,L], y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[b_size, random_dim])
            fake_labels = np.random.choice(np.arange(0.0, 1.0, 0.1), size=(b_size, 1))
            noise = tf.concat([noise, fake_labels], axis=1)
            y_gen = np.ones(b_size)
            discriminator.trainable = False
            gan_hist = gan.train_on_batch([noise, fake_labels], y_gen)

            d_l, d_a = dis_hist
            g_l, g_a = gan_hist
            
        pbar.update(1)
        pbar.set_postfix({"d_loss": "%.4f"%d_l,
                              "d_acc":"%.4f"%d_a,
                              "g_loss": "%.4f"%g_l,
                              "g_acc":"%.4f"%g_a,})
        dis_loss.append(d_l)
        dis_accuracy.append(d_a)
        gan_loss.append(g_l)
        gan_accuracy.append(g_a)
        
        #display.clear_output(wait=True)
        plot_generated_images_rad(0, generator)
    
    show_plt()
    # cgan 10*10
    plot_generated_images(generator)
    
    # cgan 10*5        
    plot_generated_images_seeds(generator)
    
def plot_generated_images_rad(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim+1])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show() #savefig('gan_generated_image_epoch_%d.png' % epoch)


def plot_generated_images(generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    fake_labels = np.empty((0, 1))
    for label in range(10):
        labels = np.random.choice(np.arange(label/10, label/10 + 0.1, 0.1), size=(10, 1))
        fake_labels = tf.concat([fake_labels, labels], axis=0)

    noise = tf.concat([noise, fake_labels], axis=1)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show() #savefig('gan_generated_image_epoch_%d.png' % epoch)

def plot_generated_images_seeds(generator, examples=50, dim=(10, 5), figsize=(10, 10)):

    seedAB = np.random.normal(0, 1, size=[2, random_dim])
    seedA = seedAB[0]
    seedB = seedAB[1]
    noise = seedA,0.75*seedA+0.25*seedB,0.5*seedA+0.5*seedB,0.25*seedA+0.75*seedB,seedB 

    tmp_noise = np.empty((0, 101))

    for label in range(10):
        labels = np.random.choice(np.arange(label/10, label/10 + 0.1, 0.1), size=(5, 1))
        tmp_noise = tf.concat([tmp_noise, tf.concat([noise, labels], axis=1)], axis=0)

    generated_images = generator.predict(tmp_noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show() #savefig('gan_generated_image_epoch_%d.png' % epoch)

def show_plt():
    plt.plot(dis_accuracy, label="dis_accuracy")
    plt.plot(gan_accuracy, label="gan_accuracy")
    plt.ylabel("Aaccuracy")
    plt.xlabel("Epochs #")
    plt.legend()
    plt.show()

    plt.plot(dis_loss, label="dis_loss")
    plt.plot(gan_loss, label="gan_loss")
    plt.ylabel("Losss")
    plt.xlabel("Epochs #")
    plt.legend()
    plt.show()

train(epochs=1)