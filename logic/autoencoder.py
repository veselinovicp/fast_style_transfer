'''Trains a denoising autoencoder on MNIST dataset.

Denoising is one of the classic applications of autoencoders.
The denoising process removes unwanted noise that corrupted the
true signal.

Noise + Data ---> Denoising Autoencoder ---> Data

Given a training dataset of corrupted data as input and
true signal as output, a denoising autoencoder can recover the
hidden structure to generate clean data.

This example has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import cv2
from skimage import feature
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class AutoEncoder:
    def __init__(self, train_size=100, epochs=100, batch_size=128, input_images='../data/michelangelo', dimension=200):
        self.train_size = train_size
        self.train_percent = 80.
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_images = input_images
        self.dimension = dimension
        self.__load_data()
        self.__prepare_model()

    def __prepare_model(self):
        np.random.seed(1337)

        x_train = self.train_output
        self.x_test = self.test_output

        self.image_size = x_train.shape[1]

        x_train_noisy = self.train_input
        self.x_test_noisy = self.test_input

        # Network parameters
        input_shape = (self.image_size, self.image_size, 1)
        batch_size = self.batch_size
        kernel_size = 3
        latent_dim = 16
        # Encoder/Decoder number of CNN layers and filters per layer
        layer_filters = [32, 64]

        # Build the Autoencoder Model
        # First build the Encoder Model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use MaxPooling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        # Shape info needed to build Decoder Model
        shape = K.int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)  # sigmoid

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        self.autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        self.autoencoder.summary()

        self.autoencoder.compile(loss='mse', optimizer='adam')

    def __load_data(self):
        print("Start loading data")

        generator = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

        train = generator.flow_from_directory(self.input_images,
                                              target_size=(self.dimension, self.dimension),
                                              batch_size=32)  # ,  class_mode='input'

        gray_scales = []
        edgeses = []
        for i in range(self.train_size):
            print((i * 100. / self.train_size), " %")
            img, _ = train.next()
            gray_scale = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
            # gray_scale = cv2.resize(gray_scale, (self.dimension, self.dimension))

            edges = feature.canny(gray_scale, sigma=3.)
            # edges = cv2.resize(edges, (200, 200))
            if i == 0:
                print("output min", img_to_array(gray_scale).min())
                print("output max", img_to_array(gray_scale).max())

            gray_scales.append(img_to_array(gray_scale))
            edgeses.append(img_to_array(edges))

            if i == 0:
                print("input min", img_to_array(edges).min())
                print("output max", img_to_array(gray_scale).max())
            # plt.imsave('../data/michelangelo/input_image'+str(i)+'.png', edges, cmap=plt.cm.gray)
            # plt.imsave('../data/michelangelo/output_image' + str(i) + '.png', gray_scale, cmap=plt.cm.gray)

        self.input_images = np.stack(edgeses)
        print("input image shape ", self.input_images.shape)
        self.output_images = np.stack(gray_scales)
        print("output image shape ", self.output_images.shape)

        split_num = int(self.train_percent / 100. * self.train_size)
        print("split_num: ", split_num)

        self.train_input = self.input_images[:split_num, :, :, :]
        print("train_input shape ", self.train_input.shape)
        self.train_output = self.output_images[:split_num, :, :, :]
        print("train_output shape ", self.train_output.shape)
        self.test_input = self.input_images[split_num:, :, :, :]
        print("test_input shape ", self.test_input.shape)
        self.test_output = self.output_images[split_num:, :, :, :]
        print("test_output shape ", self.test_output.shape)

    def predict_test_value(self, weights_path, image_path="../data/input_image.png", detect_edges=False):

        self.autoencoder.load_weights(weights_path)
        input = cv2.imread(image_path)
        input = cv2.resize(input, (self.image_size, self.image_size))
        gray_scale = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        if detect_edges:
            gray_scale = feature.canny(gray_scale, sigma=3.)
        array = img_to_array(gray_scale)
        array = array.reshape((1,) + array.shape)
        predicted_output = self.autoencoder.predict(array)
        print("shape ", predicted_output.shape)
        predicted_image = array_to_img(predicted_output[0])
        plt.imsave('../data/predicted_image.png', predicted_image, cmap=plt.cm.gray)

    def train(self):

        model_checkpoint = ModelCheckpoint("../output/weights.hdf5", monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='min')  # weights.{epoch:02d}.hdf5
        tensor_board = TensorBoard(log_dir='../output/', histogram_freq=0,
                                   write_graph=True, write_images=False)  #

        # Train the autoencoder
        self.autoencoder.fit(self.train_input,
                             self.train_output,
                             validation_data=(self.x_test_noisy, self.x_test),
                             epochs=self.epochs,
                             batch_size=self.batch_size, callbacks=[model_checkpoint, tensor_board])  # epochs=30,

        # Predict the Autoencoder output from corrupted test images
        x_decoded = self.autoencoder.predict(self.x_test_noisy)

        # Display the 1st 8 corrupted and denoised images
        # rows, cols = 10, 30
        rows, cols = 1, self.test_input.shape[0]
        num = rows * cols
        imgs = np.concatenate([self.x_test[:num], self.x_test_noisy[:num], x_decoded[:num]])
        imgs = imgs.reshape((rows * 3, cols, self.image_size, self.image_size))
        imgs = np.vstack(np.split(imgs, rows, axis=1))
        imgs = imgs.reshape((rows * 3, -1, self.image_size, self.image_size))
        imgs = np.vstack([np.hstack(i) for i in imgs])
        imgs = (imgs * 255).astype(np.uint8)
        plt.figure()
        plt.axis('off')
        plt.title('Original images: top rows, '
                  'Corrupted Input: middle rows, '
                  'Denoised Input:  third rows')
        plt.imshow(imgs, interpolation='none', cmap='gray')
        Image.fromarray(imgs).save('corrupted_and_denoised.png')
        plt.show()
