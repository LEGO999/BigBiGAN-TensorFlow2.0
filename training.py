import tensorflow as tf
import logging
import time
from losses import disc_loss, gen_en_loss
from misc import get_fixed_random, generate_images

def train(config, gen, disc_f, disc_h, disc_j, model_en, train_data):

    # Start training
    # Define optimizers
    disc_optimizer = tf.optimizers.Adam(learning_rate=config.lr_disc,
                                        beta_1=config.beta_1_disc,
                                        beta_2=config.beta_2_disc)

    gen_en_optimizer = tf.optimizers.Adam(learning_rate=config.lr_gen_en,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)


    # Define Logging to Tensorboard
    summary_writer = tf.summary.create_file_writer(f'{config.result_path}/{config.model}_{config.dataset}_{time.strftime("%Y-%m-%d--%H-%M-%S")}')

    fixed_z, fixed_c = get_fixed_random(config, num_to_generate=100)  # fixed_noise is just used for visualization.

    # Define metric
    metric_loss_gen_en = tf.keras.metrics.Mean() # Create a new instance of the class Mean
    metric_loss_disc = tf.keras.metrics.Mean()

    # Start training
    epoch_tf = tf.Variable(0, trainable=False, dtype=tf.float32)
    for epoch in range(config.num_epochs):
        logging.info(f'Start epoch {epoch+1} ...')  # logs a message.
        epoch_tf.assign(epoch)
        start_time = time.time()

        train_epoch(train_data, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
                    metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config)
        epoch_time = time.time()-start_time

        # Save results
        logging.info(f'Epoch {epoch+1}: Disc_loss: {metric_loss_disc.result()}, Gen_loss: {metric_loss_gen_en.result()}, Time: {epoch_time}')
        with summary_writer.as_default():
            tf.summary.scalar('Gen and en loss',metric_loss_gen_en.result(),step=epoch)
            tf.summary.scalar('Discriminator loss', metric_loss_disc.result(),step=epoch)

        metric_loss_gen_en.reset_states() # Resets all of the metric state variables.
        # This function is called between epochs/steps, when a metric is evaluated during training.
        metric_loss_disc.reset_states()
        # Generate images
        gen_image = generate_images(gen, fixed_z, fixed_c, config) # Using fixed_nosie to generate image.
        # Please check generate_images in misc.py
        with summary_writer.as_default():
            tf.summary.image('Generated Images', tf.expand_dims(gen_image,axis=0),step=epoch) # Write an image summary.


def train_epoch(train_data, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer,
                metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config):
    if config.conditional:
        for image, label in train_data:
            # label = tf.one_hot(label, depth=config.num_classes, dtype=tf.float32)
            train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
                       metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config)
    else:
        for image, _ in train_data:
            label = None
            train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
                       metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config)

@tf.function
def train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
               metric_loss_gen_en, batch_size, cont_dim, config):
    print('Graph will be traced...')
    with tf.device('{}:*'.format(config.device)):
        for _ in range(config.D_G_ratio):
            fake_noise = tf.random.truncated_normal([batch_size, cont_dim])
            with tf.GradientTape(persistent=True) as gen_en_tape, tf.GradientTape() as en_tape:
                fake_img = gen(fake_noise, label, training=True)
                latent_code_real = model_en(image, training=True)
                with tf.GradientTape(persistent=True) as disc_tape:
                    real_f_to_j, real_f_score = disc_f(image, label, training=True)
                    fake_f_to_j, fake_f_score = disc_f(fake_img, label, training=True)
                    real_h_to_j, real_h_score = disc_h(latent_code_real, training=True)
                    fake_h_to_j, fake_h_score = disc_h(fake_noise, training=True)
                    real_j_score = disc_j(real_f_to_j, real_h_to_j, training=True)
                    fake_j_score = disc_j(fake_f_to_j, fake_h_to_j, training=True)

                    d_loss = disc_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
                    g_e_loss = gen_en_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)

            grad_disc_f = disc_tape.gradient(d_loss, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables)

            disc_optimizer.apply_gradients(zip(grad_disc_f, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables))
            metric_loss_disc.update_state(d_loss)  # upgrade the value in metrics for single step.

        grad_gen = gen_en_tape.gradient(g_e_loss, gen.trainable_variables)

        gen_en_optimizer.apply_gradients(zip(grad_gen, gen.trainable_variables + model_en.trainable_variables))
        metric_loss_gen_en.update_state(g_e_loss)

        del gen_en_tape, en_tape
        del disc_tape




