import logging
import tensorflow_datasets as tfds
import tensorflow as tf

NUM_CALLS = tf.data.experimental.AUTOTUNE
NUM_PREFETCH = tf.data.experimental.AUTOTUNE

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # Rescale image to 32x32 if mnist/fmnist
    image = tf.image.resize(image, [32,32])
    return image, label


def get_dataset(config):
    datasets, ds_info = tfds.load(name=config.dataset, with_info=True, as_supervised=True, data_dir=config.dataset_path)
    train_data, test_data = datasets['train'], datasets['test']
    return train_data, test_data


def get_train_pipeline(dataset,config):
    dataset = dataset.map(scale, num_parallel_calls=NUM_CALLS)
    if(config.cache_dataset):
        dataset = dataset.cache()
    dataset = dataset.shuffle(config.data_buffer_size).batch(config.train_batch_size,drop_remainder=True).prefetch(NUM_PREFETCH)
    return dataset

