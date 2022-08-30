from math import dist

import numpy as np
import tensorflow as tf

points = np.array(
    [
        [-.51,  .59,  .25],
        [-.64,  .60, -.43],
        [ .76,  .48,  .35],
        [ .82,  .65, -.05],
        [-.63, -.27, -.33],
        [ .40,  .67, -.13],
    ]
)
max_d = max([dist(p, q) for p in points for q in points])


class Contextualiser(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.numpy_function(self.contextualise, [inputs], tf.double)
        return x

    @staticmethod
    def _get_weighted_midpoint(weights):
        return np.sum(points * weights[:, None], axis=0) / np.sum(weights)

    @staticmethod
    def _shift(tonal, wmp):
        return tonal - ((tonal - wmp) * .2)

    def contextualise(self, weights):
        outputs = np.zeros((len(weights), len(points)))

        for i, w in enumerate(weights):
            if i == 0:
                tonal = self._get_weighted_midpoint(w)
            else:
                wmp = self._get_weighted_midpoint(w)
                tonal = self._shift(tonal, wmp)

            norm_d = np.array([dist(tonal, p) for p in points]) / max_d
            outputs[i] = 1 - norm_d

        return outputs
