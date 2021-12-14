#!/usr/bin/env python3
"""
Create a variational autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the
        latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
            representation the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization
        and binary cross-entropy loss
    All layers should use a relu activation except for
        the mean and log variance layers in the encoder, which should use None,
        and the last layer in the decoder, which should use sigmoid
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims, ))

    encoder_hidden = keras.layers.Dense(hidden_layers[0],
                                        activation='relu')(encoder_input)

    for i in range(1, len(hidden_layers)):
        encoder_hidden = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(encoder_hidden)

    mean = keras.layers.Dense(latent_dims)(encoder_hidden)
    var = keras.layers.Dense(latent_dims)(encoder_hidden)

    def sampling(args):
        mean, var = args
        m = keras.backend.shape(mean)[0]
        dimensions = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(m, dimensions))

        return mean + keras.backend.exp(0.5 * var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([mean, var])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims, ))
    decoder_hidden = keras.layers.Dense(hidden_layers[-1],
                                        activation='relu')(decoder_input)

    for i in range(len(hidden_layers)-2, -1, -1):
        decoder_hidden = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(decoder_hidden)
    decoder_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_hidden)

    # Models
    enc = keras.models.Model(inputs=encoder_input,
                             outputs=[z, mean, var])
    dec = keras.models.Model(inputs=decoder_input,
                             outputs=decoder_output)
    # Autoencoder
    autoencoder = keras.models.Model(inputs=encoder_input,
                                     outputs=dec(enc(encoder_input)[0]))

    def loss(val1, val2):
        loss = keras.backend.binary_crossentropy(val1, val2)
        loss = keras.backend.sum(loss, axis=1)
        kl_loss = (1 + var - keras.backend.square(mean)
                   - keras.backend.exp(var))
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)

        return loss + kl_loss

    autoencoder.compile(optimizer='adam', loss=loss)

    return enc, dec, autoencoder
