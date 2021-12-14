#!/usr/bin/env python3
"""
Create a convolutional autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a Convolutional autoencoder:

    input_dims is a tuple of integers containing the dimensions of
        the model input
    filters is a list containing the number of filters for each
        convolutional layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions
        of the latent space representation
    Each convolution in the encoder should use a kernel size of (3, 3)
        with same padding and relu activation,
        followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should
        use a filter size of (3, 3) with same padding and relu activation,
        followed by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as
            the number of channels in input_dims with sigmoid
            activation and no upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using
        adam optimization and binary cross-entropy loss
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    encoder_hidden = keras.layers.Conv2D(filters=filters[0],
                                         kernel_size=3,
                                         padding='same',
                                         activation='relu')(encoder_input)
    encoder_hidden = keras.layers.MaxPool2D(pool_size=(2, 2),
                                            padding='same')(encoder_hidden)
    for i in range(1, len(filters)):
        encoder_hidden = keras.layers.Conv2D(filters=filters[i],
                                             kernel_size=3,
                                             padding='same',
                                             activation='relu')(encoder_hidden)
        encoder_hidden = keras.layers.MaxPool2D(pool_size=(2, 2),
                                                padding='same')(encoder_hidden)
    encoder_output = encoder_hidden

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    decoder_hidden = keras.layers.Conv2D(filters=filters[-1],
                                         kernel_size=3,
                                         padding='same',
                                         activation='relu')(decoder_input)
    decoder_hidden = keras.layers.UpSampling2D(2)(decoder_hidden)
    for i in range(len(filters) - 2, 0, -1):
        decoder_hidden = keras.layers.Conv2D(filters=filters[i],
                                             kernel_size=3,
                                             padding='same',
                                             activation='relu')(decoder_hidden)
        decoder_hidden = keras.layers.UpSampling2D(2)(decoder_hidden)
    decoder_hidden = keras.layers.Conv2D(filters=filters[0],
                                         kernel_size=3,
                                         padding='valid',
                                         activation='relu')(decoder_hidden)
    decoder_hidden = keras.layers.UpSampling2D(2)(decoder_hidden)
    decoder_output = keras.layers.Conv2D(filters=input_dims[-1],
                                         kernel_size=3,
                                         padding='same',
                                         activation='sigmoid')(decoder_hidden)

    # Models
    encoder = keras.models.Model(inputs=encoder_input, outputs=encoder_output)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)
    autoencoder = keras.models.Model(inputs=encoder_input,
                                     outputs=decoder(encoder(encoder_input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
