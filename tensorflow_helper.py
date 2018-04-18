import tensorflow as tf
import numpy as np


class TfHelper:
    """
    Class for quick generation of neural networks in tensorflow
    """

    @staticmethod
    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape=shape,
                                               stddev=0.05))

    @staticmethod
    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    @staticmethod
    def new_conv_layer(input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       use_pooling=True,  # Use 2x2 max-pooling.
                       strides=(1, 1, 1, 1)): # Strides

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = TfHelper.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = TfHelper.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv3d(input=input,
                             filter=weights,
                             strides=list(strides),
                             padding="SAME")

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME")

        # ADD a Relu
        tf.nn.relu(layer)

        return layer, weights

    @staticmethod
    def flatten_layer(layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    @staticmethod
    def new_fc_layer(input,          # The previous layer.
                    num_inputs,     # Num. inputs from prev. layer.
                    num_outputs,    # Num. outputs.
                    use_relu=True): # Use Rectified Linear Unit (ReLU)?

        weights = TfHelper.new_weights(shape=[num_inputs, num_outputs])
        biases = TfHelper.new_biases(length=num_outputs)

        layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)

        return layer
