import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E

def set_up_train(config):
    # Setup tensorflow
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data, _ = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)

    # get model
    logging.info('Prepare model for training...')

    model_generator = BIGBIGAN_G(config)
    model_discriminator_f = BIGBIGAN_D_F(config)
    model_discriminator_h = BIGBIGAN_D_H(config)
    model_discriminator_j = BIGBIGAN_D_J(config)
    model_encoder = BIGBIGAN_E(config)

    # train
    logging.info('Start training...')
    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data)
    # Finished
    logging.info('Training finished ;)')
