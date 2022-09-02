import tensorflow as tf
from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric


def precision(class_no):
    def wrapper(y_true, y_pred):
        # Transform labels and logits into workable tensors.
        y_true = tf.reshape(tf.transpose(y_true), [y_true.shape[0]])
        y_pred = tf.argmax(y_pred, axis=1)

        # Calculate tp and fp rates for class.
        tp = tf.math.count_nonzero((y_pred == class_no) & (y_true == class_no))
        fp = tf.math.count_nonzero((y_pred == class_no) & (y_true != class_no))
        total = tp + fp

        # Return result. This contains a fallback to set precision to 0
        # rather than NaN if the net predicts no positives at all.
        if tf.cond(total == 0, lambda: True, lambda: False):
            return tf.constant(0., dtype=tf.float64)
        return tp / total

    return wrapper


class PrecisionForClass(base_metric.MeanMetricWrapper):
    @dtensor_utils.inject_mesh
    def __init__(self, class_no, name="precision", dtype=None):
        super().__init__(precision(class_no), f"{name}_{class_no}", dtype=dtype)


def recall(class_no):
    def wrapper(y_true, y_pred):
        # Transform labels and logits into workable tensors.
        y_true = tf.reshape(tf.transpose(y_true), [y_true.shape[0]])
        y_pred = tf.argmax(y_pred, axis=1)

        # Calculate tp and fn rates for class.
        tp = tf.math.count_nonzero((y_pred == class_no) & (y_true == class_no))
        fn = tf.math.count_nonzero((y_pred != class_no) & (y_true == class_no))
        total = tp + fn

        # Return result. This contains a fallback to set recall to 0
        # rather than NaN if the net predicts no positives at all.
        if tf.cond(total == 0, lambda: True, lambda: False):
            return tf.constant(0., dtype=tf.float64)
        return tp / total

    return wrapper


class RecallForClass(base_metric.MeanMetricWrapper):
    @dtensor_utils.inject_mesh
    def __init__(self, class_no, name="recall", dtype=None):
        super().__init__(recall(class_no), f"{name}_{class_no}", dtype=dtype)


def mcc(class_no):
    def wrapper(y_true, y_pred):
        # Transform labels and logits into workable tensors.
        y_true = tf.reshape(tf.transpose(y_true), [y_true.shape[0]])
        y_pred = tf.argmax(y_pred, axis=1)

        # Calculate tp, fp, tn, and fp rates for class.
        tp = tf.math.count_nonzero((y_pred == class_no) & (y_true == class_no))
        tn = tf.math.count_nonzero((y_pred != class_no) & (y_true != class_no))
        fp = tf.math.count_nonzero((y_pred == class_no) & (y_true != class_no))
        fn = tf.math.count_nonzero((y_pred != class_no) & (y_true == class_no))
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Return result. This contains a fallback to set MCC to 0 rather
        # than NaN if the coefficient cannot be determined.
        if tf.cond(denom == 0, lambda: True, lambda: False):
            return tf.constant(0., dtype=tf.float64)
        return tf.divide(
            tf.cast(tp * tn - fp * fn, tf.float64),
            tf.sqrt(tf.cast((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), tf.float64)),
        )

    return wrapper


class MccForClass(base_metric.MeanMetricWrapper):
    @dtensor_utils.inject_mesh
    def __init__(self, class_no, name="mcc", dtype=None):
        super().__init__(mcc(class_no), f"{name}_{class_no}", dtype=dtype)


def f1(class_no):
    def wrapper(y_true, y_pred):
        # Get precision and recall.
        p = precision(class_no)(y_true, y_pred)
        r = recall(class_no)(y_true, y_pred)
        total = p + r

        # Return result. This contains a fallback to set F1 to 0 rather
        # than NaN if the precision and the recall are both 0.
        if tf.cond(total == 0, lambda: True, lambda: False):
            return tf.constant(0., dtype=tf.float64)
        return (2 * p * r) / (p + r)

    return wrapper


class F1ForClass(base_metric.MeanMetricWrapper):
    @dtensor_utils.inject_mesh
    def __init__(self, class_no, name="f1", dtype=None):
        super().__init__(f1(class_no), f"{name}_{class_no}", dtype=dtype)
