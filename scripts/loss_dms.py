import tensorflow as tf


def MAE(T, P):
    if P.shape.rank == 3:  # Check the rank of the tensor
        P = tf.reshape(P, (tf.shape(P)[0], 1, tf.shape(P)[1], tf.shape(P)[2]))
    W = tf.ones_like(T)
    Numerator = W * tf.abs(P - T)

    # Ensure we squeeze the correct axis
    if Numerator.shape.rank == 4 and Numerator.shape[1] == 1:  # Check if the second dimension is 1
        Numerator = tf.squeeze(Numerator, axis=1)
        W = tf.squeeze(W, axis=1)

    sum_Numerator = tf.reduce_sum(tf.reshape(Numerator, [tf.shape(Numerator)[0], -1]), axis=1)
    sum_W = tf.reduce_sum(tf.reshape(W, [tf.shape(W)[0], -1]), axis=1)

    return sum_Numerator / sum_W

class MAELoss(tf.keras.losses.Loss):
    def call(self, T, P):
        return tf.reduce_sum(MAE(T, P))

class DMSWMSELoss(tf.keras.losses.Loss):
    def call(self, T, P):
        # Calculate the weight for each pixel
        max_truth = tf.reduce_max(tf.reshape(T, [tf.shape(T)[0], -1]), axis=1, keepdims=True)
        max_truth = tf.reshape(max_truth, [-1, 1, 1, 1])
        weight = 1. + tf.abs(T) / max_truth
        # Calculate the weighted MSE
        wmse = weight * tf.square(P - T)
        loss = tf.reduce_sum(wmse, axis=(1, 2, 3))
        return tf.reduce_sum(loss)

class DMSLoss(tf.keras.losses.Loss):
    def __init__(self, eta_1=1.0, eta_2=1e-3):
        super(DMSLoss, self).__init__()
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.mae_loss = MAELoss()
        self.dmswmse_loss = DMSWMSELoss()

    def call(self, T, P):
        mae_val = self.eta_1 * self.mae_loss(T, P)
        dmswmse_val = self.eta_2 * self.dmswmse_loss(T, P)
        return mae_val + dmswmse_val