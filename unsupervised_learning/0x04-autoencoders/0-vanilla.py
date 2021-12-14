#!/usr/bin/env python3
"""
Create a Vanilla autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
            layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
            representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    All layers should use a relu activation except for the last layer
        in the decoder, which should use sigmoid
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims, ))
    encoder_hidden = keras.layers.Dense(hidden_layers[0],
                                        activation='relu')(encoder_input)
    for i in range(1, len(hidden_layers)):
        encoder_hidden = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(encoder_hidden)
    encoder_output = keras.layers.Dense(latent_dims,
                                        activation='relu')(encoder_hidden)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims, ))
    decoder_hiddden = keras.layers.Dense(hidden_layers[-1],
                                         activation='relu')(decoder_input)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoder_hidden = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(decoder_hiddden)
    decoder_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_hidden)

    # Models
    encoder = keras.models.Model(inputs=encoder_input, outputs=encoder_output)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)
    autoencoder = keras.models.Model(inputs=encoder_input,
                                     outputs=decoder(encoder(encoder_input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
