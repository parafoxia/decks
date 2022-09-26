from math import dist

import numpy as np
import tensorflow as tf


class Contextualiser(tf.keras.layers.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.points = np.array(
            [
                [-0.666,  0.730,  0.314],
                [-0.854,  0.680, -0.414],
                [ 0.960,  0.648,  0.588],
                [ 1.000,  0.038,  0.346],
                [-0.896, -0.424, -0.672],
                [ 0.750,  0.750,  0.124],
            ]
        )
        self.max_d = max([dist(p, q) for p in self.points for q in self.points])
        self.alpha = alpha

    @property
    def path(self):
        return getattr(self, "_path", None)

    def call(self, inputs):
        x = tf.numpy_function(self.contextualise, [inputs], tf.double)
        return x

    def _get_weighted_midpoint(self, weights):
        return np.sum(self.points * weights[:, None], axis=0) / np.sum(weights)

    def _shift(self, tonal, wmp):
        return tonal - ((tonal - wmp) * self.alpha)

    def contextualise(self, weights):
        outputs = np.empty((len(weights), len(self.points)))
        self._path = np.empty((len(weights), 3))

        for i, w in enumerate(weights):
            if i == 0:
                tonal = self._get_weighted_midpoint(w)
            else:
                wmp = self._get_weighted_midpoint(w)
                tonal = self._shift(tonal, wmp)

            self._path[i, :] = tonal
            norm_d = np.array([dist(tonal, p) for p in self.points]) / self.max_d
            outputs[i] = 1 - norm_d

        return outputs
