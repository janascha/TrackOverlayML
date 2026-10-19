import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="TrackOverlayML")
class BalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.pos = self.add_weight(name="pos", initializer="zeros")
        self.neg = self.add_weight(name="neg", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

        y_pred = tf.cast(
            tf.reshape(y_pred, [-1]) >= 0.5,
            tf.float32
        )

        self.tp.assign_add(
            tf.reduce_sum(y_true * y_pred)
        )

        self.tn.assign_add(
            tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
        )

        self.pos.assign_add(
            tf.reduce_sum(y_true)
        )

        self.neg.assign_add(
            tf.reduce_sum(1.0 - y_true)
        )

    def result(self):

        diff_recall = tf.math.divide_no_nan(
            self.tp,
            self.pos
        )

        nodiff_recall = tf.math.divide_no_nan(
            self.tn,
            self.neg
        )

        return 0.5 * (diff_recall + nodiff_recall)

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.pos.assign(0.0)
        self.neg.assign(0.0)
