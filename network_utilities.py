import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras.models import load_model
from PIL import Image
from image_utilities import ImageUtilities

np.random.seed(1337)  # for reproducibility

class NetworkUtils:

    BASE_DIRECTORY = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/"

    @staticmethod
    def keras_cnn(input_image_directory: str, output_image_directory: str, train_percentage=0.70,
                  nb_epoch=30, batch_size=32, img_size=(64, 64), strides=(1, 1, 1), kernel_size=(3, 3, 1),
                  nb_filters=32, pool_size=(2, 2), dropout=0.25, plot=True, verbose_training=1):
        """
        :param input_image_directory: input image directory path (str)
        :param output_image_directory: desired image directory path (str)
        :param train_percentage: what percentage of the total data do you want in training?
        :param nb_epoch: number of epochs while training
        :param batch_size: neural network training batch size
        :param img_size: tuple in the form of (height, width) of desired image size
        :param strides: tuple for the stride in a 2d convolution
        :param kernel_size: tuple for size of a 2d convolutional kernel
        :param nb_filters: number of convolutional filters
        :param pool_size: tuple of size for max-pooling
        :param dropout: percentage of dropout neurons (0-1)
        :param plot: whether to save an image of the graph of the loss function over time
        :param verbose_training: whether to display the progress of the neural network training
        """

        # input image dimensions
        img_rows, img_cols = img_size[0], img_size[1]

        # number of convolutional filters to use
        # size of pooling area for max pooling
        # convolution kernel size

        X_Data = ImageUtilities.load_images_as_np_array(input_image_directory, verbose=False, flatten=True,
                                                        width=img_cols, height=img_rows)
        Y_Data = ImageUtilities.load_images_as_np_array(output_image_directory, verbose=False, flatten=True,
                                                        width=img_cols, height=img_rows)

        # Split the data into training and test sets
        num_training = int(X_Data.shape[0] * train_percentage)

        X_train = X_Data[0:num_training]
        X_test = X_Data[num_training:X_Data.shape[0]]

        Y_train = Y_Data[0:num_training]
        Y_test = Y_Data[num_training:Y_Data.shape[0]]
        Y_train = Y_train.reshape(Y_train.shape[0], img_rows, img_cols, 1, 1)
        Y_test = Y_test.reshape(Y_test.shape[0], img_rows, img_cols, 1, 1)
        print("Y_train Shape:", Y_train.shape)

        print("Data split into Training and Testing\n")
        # X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1, 1)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples\n')

        nb_classes = Y_train.shape[1]
        print("total_classes:", nb_classes)

        input_shape = (img_rows, img_cols, 1, 1)   # input shape is: (height, width, 1, 1)
        print("input shape:", input_shape)
        # exit(0)

        model = Sequential()

        # Layer 1: no batch normalization
        model.add(Conv3D(1, kernel_size=kernel_size, padding="same",
                         input_shape=input_shape, strides=strides))
        model.add(Activation('relu'))

        filter_size = list(kernel_size[:-1])
        filter_size.append(nb_filters)
        filter_size = tuple(filter_size)
        print("filter Size:", filter_size)

        input_shape = (img_rows, img_cols, 1, nb_filters)

        # Layer 2
        model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
                         input_shape=input_shape, strides=strides))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # Layer 3
        model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
                         input_shape=input_shape, strides=strides))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # Layer 4
        # model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
        #                  input_shape=input_shape, strides=strides))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))

        # Layer 5
        # model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
        #                  input_shape=input_shape, strides=strides))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))

        # Layer 6
        # model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
        #                  input_shape=input_shape, strides=strides))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))

        # Layer 7
        # model.add(Conv3D(nb_filters, kernel_size=filter_size, padding="same",
        #                  input_shape=input_shape, strides=strides))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))

        # Layer 8
        model.add(Conv3D(1, kernel_size=filter_size, padding="same",
                         input_shape=input_shape, strides=strides))
        model.add(Activation("relu"))

       # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=pool_size))
        # model.add(Dropout(dropout))
        #
        # model.add(Flatten())
        # model.add(Dense(2048))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout*2))
        # model.add(Dense(nb_classes))

        start = time.time()

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                            verbose=verbose_training, validation_data=(X_test, Y_test))

        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        model.save("/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/models/keras_CNN_" + timestr)

        end = time.time()
        print("Model took %0.2f seconds to train" % (end - start))

        score = model.evaluate(X_test, Y_test, verbose=0)
        """Start Generating Metrics"""

        if plot:

            graph_directory = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/graphs"
            # Plot the Loss Curves
            plt.figure(figsize=[8, 6])
            plt.plot(history.history['loss'], 'r', linewidth=3.0)
            plt.plot(history.history['val_loss'], 'b', linewidth=3.0)

            plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
            plt.xlabel('Epochs ', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.title('Loss Curves', fontsize=16)

            # Save the Graph png and overwrite the latest graph
            loss_curve_name = "Loss_Graph_" + timestr + ".png"
            latest_curve_name = os.path.join(graph_directory, loss_curve_name)
            plt.savefig(latest_curve_name)

            print("Plotting complete")

        print('Test score:', score[0])

    @staticmethod
    def neural_network_tf(inputs: str, outputs: str, training_size=0.7, type):

        print("Loading Data to run a Tensorflow Neural Network...\n======\n")
        try:
            x_data = ImageUtilities.load_images_as_np_array(os.path.join(BASE_DIRECTORY, inputs))
            y_data = ImageUtilities.load_images_as_np_array(os.path.join(BASE_DIRECTORY, outputs))

        except FileNotFoundError as e:
            print("Error finding directories: ")
            print("Input Directory: ", inputs)
            print("Output Directory:", outputs)
            exit(1)

        print("Data Loaded")
        print("Splitting data...")
        print("======\n")

        x_train = x_data[0: int(x_data.shape[0] * training_size)]
        x_test = x_data[int(x_data.shape[0] * training_size):]

        y_train = y_data[0: int(y_data.shape[0] * training_size)]
        y_test = y_data[int(y_data.shape[0] * training_size):]

        print("Size of:")
        print("======")
        print("- Training-set:\t\t{}".format(x_train.shape[0]))
        print("- Test-set:\t\t\t{}\n".format(x_test.shape[0]))




    @staticmethod
    def denoise_images(model_file: str, input_image_directory: str, output_image_directory: str, new_image_directory,
                       width=64, height=64):

        print("Denoising Images...")
        model = load_model(model_file)

        input_array = ImageUtilities.load_images_as_np_array(image_directory=input_image_directory, flatten=False,
                                                             width=width, height=height, verbose=False)
        output_array = ImageUtilities.load_images_as_np_array(image_directory=output_directory, flatten=False,
                                                              width=width, height=height, verbose=False)
        output_image = output_array[1:2]
        input_image = input_array[1:2]
        image_array = input_image.reshape(1, width, height, 1, 1)

        prediction = model.predict(x=image_array)

        prediction = prediction.reshape(width, height)
        prediction = ImageUtilities.convert_to_uint8(prediction)
        prediction = Image.fromarray(prediction)
        prediction.save("prediction.jpg", "JPEG")

        input_image = input_image.reshape(width, height)
        input_image = ImageUtilities.convert_to_uint8(input_image)
        input_image = Image.fromarray(input_image)
        input_image.save("Input Image.jpg", "JPEG")

        output_image = output_image.reshape(width, height)
        output_image = ImageUtilities.convert_to_uint8(output_image)
        output_image = Image.fromarray(output_image)
        output_image.save("Output Image.jpg", "JPEG")

        path = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/models"

        # TODO: FIX The fact that there is a wrong image being displayed, linear algebra is bad..

        output_image.show("output Image")
        input_image.show("input Image")
        prediction.show("prediction")

        # output_image.save(os.path.join(path, "output image"), "JPEG")
        # input_image.save(os.path.join(path, "input image"), "JPEG")
        # prediction.save(os.path.join(path, "prediction image"), "JPEG")


        # for i in range(image_array.shape[0]):
        #     image_array[i] = image_array[i].reshape(1, 64, 64, 1)
        #     print("index:", i, ":", image_array[i].shape)
        #
        #     print(model.predict(x=image_array))
        #     #
        #     # print(np.array(image_array[i:i+1]).shape)
        #     # print(model.predict(np.array([image_array[i:i+1]])))


if __name__ == "__main__":
    BASE_DIRECTORY = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/"

    input_directory = os.path.join(BASE_DIRECTORY, "inputs")
    output_directory = os.path.join(BASE_DIRECTORY, "outputs")
    print("running commands")
    # NetworkUtils.train_neural_network(input_image_directory=input_directory, output_image_directory=output_directory)

    # NetworkUtils.keras_cnn(input_image_directory=input_directory, output_image_directory=output_directory,
    #                            nb_epoch=30, verbose_training=1)

    # latest = sorted(os.listdir("/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/models"))[-1]
    # model_path = os.path.join("/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/models", latest)

    # NetworkUtils.denoise_images(model_path,
    #                             input_image_directory=input_directory,
    #                             output_image_directory=output_directory,
    #                             new_image_directory="hi")

    NetworkUtils.neural_network_tf(input_directory, output_directory)
