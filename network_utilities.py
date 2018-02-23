import tensorflow as tf
import numpy as np
import os
import cv2
import errno
import shutil
from PIL import Image


class NetworkUtils:

    @staticmethod
    def load_images_as_np_array(image_directory: str)->np.array:
        """
        :param image_directory: directory of images to load

        :return: the images as np_array
        """

        list_of_images = []

        for image in os.listdir(image_directory):
            if image.endswith(".png") or image.endswith(".jpg"):
                image = os.path.join(image_directory, image)
                print(image)

                array = cv2.imread(image)
                list_of_images.append(array.flatten())

        train_data = np.array(list_of_images)
        return train_data

    @staticmethod
    def train_neural_network(input_image_directory: str, output_image_directory: str, train_percentage=0.70,
                             epochs=30, batch_size=32):
        """
        :param input_image_directory: String of path to input image directory
        :param output_image_directory: String of path to output image directory
        :param train_percentage: what percentage of the data do you want to train on?
        :param epochs: how many epochs for the network to train on
        :param batch_size: how big to make each batch that the network trains on
        :return:
        """
        X_Data = NetworkUtils.load_images_as_np_array(input_image_directory)
        Y_Data = NetworkUtils.load_images_as_np_array(output_image_directory)

        num_training = int(X_Data.shape[0] * train_percentage)

        X_train = X_Data[0:num_training]
        X_test = X_Data[num_training:X_Data.shape[0]]

        Y_train = Y_Data[0:num_training]
        Y_test = Y_Data[num_training:Y_Data.shape[0]]

        x = tf.placeholder("float", [None, X_train[0].shape])
        y = tf.placeholder("float", [None, Y_train[0].shape])

        prediction = NetworkUtils.neural_network_model(X_train, Y_train)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(epochs):
                epoch_loss = 0

                for _ in range(int(X_train.shape[0]/batch_size)):
                    x_epoch, y_epoch = 0, 0
                    # TODO: FIX THIS: (find sendex vid where he rewrites that function (for neg/pos text) (time: 5
                    # TODO: https://www.youtube.com/watch?v=6rDWwL6irG0&index=7&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP
                    _, epoch_cost = sess.run([optimizer, cost], feed_dict={x: x_epoch, y: y_epoch})
                    epoch_loss += epoch_cost
                print("Epoch", epoch+1, "completed out of", epochs, "loss:", epoch_loss)

            correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))

    @staticmethod
    def neural_network_model(X_train, Y_train):
        """
        :param X_train: np array of the input training data
        :param Y_train: np array of the output of the training data
        :return: the output of the computational graph after one iteration of the network
        """

        num_nodes = {
            "layer 1": 500,
            "layer 2": 500,
            "layer 3": 500
        }

        # Create the layers of the neural Network
        hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([X_train[0].shape[0], num_nodes["layer 1"]])),
                          "biases": tf.Variable(tf.random_normal(num_nodes["layer 1"]))}

        hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([num_nodes["layer 1"], num_nodes["layer 2"]])),
                          "biases": tf.Variable(tf.random_normal(num_nodes["layer 2"]))}

        hidden_layer_3 = {"weights": tf.Variable(tf.random_normal([num_nodes["layer 2"], num_nodes["layer 3"]])),
                          "biases": tf.Variable(tf.random_normal(num_nodes["layer 3"]))}

        output_layer = {"weights": tf.Variable(tf.random_normal([num_nodes["layer 3"], Y_train[0].shape[0]])),
                        "biases": tf.Variable(tf.random_normal(num_nodes[Y_train[0].shape[0]]))}

        # Assemble the Network
        layer_1 = tf.add(tf.matmul(X_train, hidden_layer_1["weights"]), hidden_layer_1["biases"])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2["weights"]), hidden_layer_2["biases"])
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3["weights"]), hidden_layer_3["biases"])
        layer_3 = tf.nn.relu(layer_3)

        output = tf.add(tf.matmul(layer_3, output_layer["weights"]), output_layer["biases"])

        return output


if __name__ == "__main__":

    X_data = NetworkUtils.load_images_as_np_array(
        "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/inputs")

    Y_data = NetworkUtils.load_images_as_np_array(
        "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/outputs")


