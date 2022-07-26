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


class DECKS(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.numpy_function(self.contextualise, [inputs], tf.double)
        return x

    @staticmethod
    def _get_weighted_midpoint(x):
        return np.sum(points * x[:, None], axis=0) / np.sum(x)

    @staticmethod
    def _shift(a, b):
        return a - ((a - b) * .2)

    def contextualise(self, x):
        outputs = np.zeros((len(x), len(points)))

        for i, pred in enumerate(x):
            if i == 0:
                dynamic = self._get_weighted_midpoint(x[0])
            else:
                wmp = self._get_weighted_midpoint(pred)
                dynamic = self._shift(dynamic, wmp)

            norm_d = np.array([dist(dynamic, p) for p in points]) / max_d
            outputs[i] = 1 - norm_d

        return outputs
