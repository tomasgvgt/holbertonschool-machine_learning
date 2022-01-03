#!/usr/bin/env python3
"""
Convert a gensim word2vec model to a keras embedding layer
"""
from gensim.models import Word2Vec
import tensorflow.keras as K


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer:

    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    keras_embedding = model.wv.get_keras_embedding(train_embeddings=True)
    return keras_embedding
