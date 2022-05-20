"""tensorflow implementation of Darknet-19 model"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Darknet19(object):
    """Implementation of Darknet-19 model"""

    def __init__(
        self, image_size, grid_size, n_anchor, n_classes, weight_file="yolov2.weights"
    ):
        """Construct a new instance

        (use https://pjreddie.com/media/files/yolov2.weights for weights)

        :arg image_size: size of input images which are of shape (image_size,image_size)
        :arg grid_size: number of grid cells in one direction, grid is of shape
                        (grid_size,grid_size)
        :arg n_anchor: number of anchor boxes
        :arg n_classes: number of different classes
        :arg weight_file: file with weights
        """
        self.image_size = image_size
        self.grid_size = grid_size
        self.n_anchor = n_anchor
        self.n_classes = n_classes
        self.true_box_buffer = 32
        self._build_model()
        self._set_weights(weight_file)

    @property
    def model(self):
        """Return model property"""
        return self.__model

    def _build_model(self):
        """Build model architecture"""
        # Specifications of layers 1 - 20
        layer_specs = [
            {"features": 32, "mask": (3, 3), "pool": True},  # layer  1
            {"features": 64, "mask": (3, 3), "pool": True},  # layer  2
            {"features": 128, "mask": (3, 3), "pool": False},  # layer  3
            {"features": 64, "mask": (1, 1), "pool": False},  # layer  4
            {"features": 128, "mask": (3, 3), "pool": True},  # layer  5
            {"features": 256, "mask": (3, 3), "pool": False},  # layer  6
            {"features": 128, "mask": (1, 1), "pool": False},  # layer  7
            {"features": 256, "mask": (3, 3), "pool": True},  # layer  8
            {"features": 512, "mask": (3, 3), "pool": False},  # layer  9
            {"features": 256, "mask": (1, 1), "pool": False},  # layer 10
            {"features": 512, "mask": (3, 3), "pool": False},  # layer 11
            {"features": 256, "mask": (1, 1), "pool": False},  # layer 12
            {"features": 512, "mask": (3, 3), "pool": True},  # layer 13
            {"features": 1024, "mask": (3, 3), "pool": False},  # layer 14
            {"features": 512, "mask": (1, 1), "pool": False},  # layer 15
            {"features": 1024, "mask": (3, 3), "pool": False},  # layer 16
            {"features": 512, "mask": (1, 1), "pool": False},  # layer 17
            {"features": 1024, "mask": (3, 3), "pool": False},  # layer 18
            {"features": 1024, "mask": (3, 3), "pool": False},  # layer 19
            {"features": 1024, "mask": (3, 3), "pool": False},  # layer 20
        ]
        input_image = keras.layers.Input(
            shape=(self.image_size, self.image_size, 3), name="input"
        )
        # ==== Layers 1 - 20 ====
        x = input_image
        for j, layer_spec in enumerate(layer_specs):
            layer_id = f"{j+1:02d}"
            x = keras.layers.Conv2D(
                layer_spec["features"],
                layer_spec["mask"],
                strides=(1, 1),
                padding="same",
                name=f"conv_{layer_id}",
                use_bias=False,
            )(x)
            x = keras.layers.BatchNormalization(name=f"norm_{layer_id}")(x)
            x = keras.layers.LeakyReLU(alpha=0.1, name=f"LReLU_{layer_id}")(x)
            # Save skip connection after layer 13, before max-pooling
            if j + 1 == 13:
                skip_connection = x
            if layer_spec["pool"]:
                x = keras.layers.MaxPooling2D(
                    pool_size=(2, 2), name=f"maxpool_{layer_id}"
                )(x)

        # ==== Layer 21 ====
        skip_connection = keras.layers.Conv2D(
            64, (1, 1), strides=(1, 1), padding="same", name="conv_21", use_bias=False
        )(skip_connection)
        skip_connection = keras.layers.BatchNormalization(name="norm_21")(
            skip_connection
        )
        skip_connection = keras.layers.LeakyReLU(alpha=0.1, name="LReLu_21")(
            skip_connection
        )
        skip_connection = keras.layers.Lambda(
            lambda z: tf.nn.space_to_depth(z, block_size=2), name="lambda_21"
        )(skip_connection)
        x = keras.layers.concatenate([skip_connection, x])
        # ==== Layer 22 ====
        x = keras.layers.Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_22", use_bias=False
        )(x)
        x = keras.layers.BatchNormalization(name="norm_22")(x)
        x = keras.layers.LeakyReLU(alpha=0.1, name="LReLU_22")(x)
        # ==== Layer 23 =====
        x = keras.layers.Conv2D(
            self.n_anchor * (4 + 1 + self.n_classes),
            (1, 1),
            strides=(1, 1),
            padding="same",
            name="conv_23",
        )(x)
        output_layer = keras.layers.Reshape(
            (self.grid_size, self.grid_size, self.n_anchor, 4 + 1 + self.n_classes)
        )(x)
        use_true_box_buffer = False
        if use_true_box_buffer:
            true_boxes = keras.layers.Input(shape=(1, 1, 1, self.true_box_buffer, 4))
            inputs = [input, true_boxes]
            outputs = keras.layers.Lambda(lambda args: args[0])(
                [output_layer, true_boxes]
            )
        else:
            inputs = input_image
            outputs = output_layer
        # Build model
        self.__model = keras.Model(inputs=inputs, outputs=outputs)

    def _set_weights(self, weight_file):
        """Read weights from disk and set weights of final convolutional to random values.

        This is essentially the code given at https://www.maskaravivek.com/post/yolov2/

        :arg weight_file: file with weights
        """

        class WeightReader(object):
            """Class for reading weights from file"""

            def __init__(self, weight_file):
                """Create new instance

                Reads all weights into a numpy array and then allows accessing
                them through slicing.

                :arg weight_file: name of file with weights
                """
                self.all_weights = np.fromfile(weight_file, dtype=np.float32)
                self.offset = 4

            def get(self, n_data):
                """Access next n_data entries of the all_weights array

                :arg n_data: number of entries to read
                """
                self.offset += n_data
                return self.all_weights[self.offset - n_data : self.offset]

        weight_reader = WeightReader(weight_file)

        nb_conv = 23  # number of convolutional layers
        for j in range(1, nb_conv + 1):
            if j < nb_conv:
                # --- batch normalisation layer ---
                norm_layer = self.model.get_layer(f"norm_{j:02d}")
                # set beta, gamma, mean and variance
                new_weights = []
                for weights in norm_layer.get_weights():
                    size = np.prod(weights.shape)
                    new_weights.append(weight_reader.get(size))
                beta, gamma, mean, var = new_weights
                norm_layer.set_weights([gamma, beta, mean, var])

            # --- convolutional layer ---
            conv_layer = self.model.get_layer(f"conv_{j:02d}")
            weights = conv_layer.get_weights()
            if len(weights) > 1:
                # read bias term
                bias_size = np.prod(weights[1].shape)
                bias = weight_reader.get(bias_size)
            # read kernel
            kernel_size = np.prod(weights[0].shape)
            kernel = weight_reader.get(kernel_size)
            kernel = kernel.reshape(list(reversed(weights[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            new_weights = [kernel, bias] if len(weights) > 1 else [kernel]
            conv_layer.set_weights(new_weights)
        # set weights of final convolutional layer to random values
        layer = self.model.get_layer(f"conv_23")
        weights = layer.get_weights()
        new_kernel = np.random.normal(size=weights[0].shape) / (
            self.grid_size * self.grid_size
        )
        new_bias = np.random.normal(size=weights[1].shape) / (
            self.grid_size * self.grid_size
        )
        layer.set_weights([new_kernel, new_bias])
