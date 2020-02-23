import tensorflow as tf

def disc_loss(real_f_output, real_h_output, real_j_output, fake_f_output, fake_h_output, fake_j_output):

    real_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(real_f_output) - real_f_output)
                               + tf.nn.relu(tf.ones_like(real_h_output) - real_h_output)
                               + tf.nn.relu(tf.ones_like(real_j_output) - real_j_output))
    fake_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(fake_f_output) + fake_f_output)
                               + tf.nn.relu(tf.ones_like(fake_h_output) + fake_h_output)
                               + tf.nn.relu(tf.ones_like(fake_j_output) + fake_j_output))
    total_loss = real_loss + fake_loss
    return total_loss


def gen_en_loss(real_f_output, real_h_output, real_j_output, fake_f_output, fake_h_output, fake_j_output):

    real_loss = tf.reduce_mean(tf.reduce_sum([real_f_output, real_h_output, real_j_output], axis=0))
    fake_loss = tf.reduce_mean((-1) * tf.reduce_sum([fake_f_output, fake_h_output, fake_j_output], axis=0))

    return real_loss + fake_loss

