import tensorflow as tf

# Compute total variation regularization loss term given a variable image (x) and its shape
def TotalVariation_loss(x, shape):
    TOTAL_VARIATION_SMOOTHING = 1.5
    with tf.name_scope('get_total_variation'):
        # Get the dimensions of the variable image
        height = shape[1]
        width = shape[2]
        size = reduce(lambda a, b: a * b, shape) ** 2

        # Disjoin the variable image and evaluate the total variation
        x_cropped = x[:, :height - 1, :width - 1, :]
        left_term = tf.square(x[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(x[:, :height - 1, 1:, :] - x_cropped)
        smoothed_terms = tf.sqrt(left_term + right_term)
        return tf.reduce_sum(smoothed_terms) / size



