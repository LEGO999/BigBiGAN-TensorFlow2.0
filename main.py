from absl import app
from absl import flags
import os
from train import set_up_train
FLAGS = flags.FLAGS



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


flags.DEFINE_string("model", 'BigBiGAN', 'Model to use (BigBiGAN|')
flags.DEFINE_string("dataset",'cifar10','Dataset (mnist|fashion_mnist|cifar10)')
flags.DEFINE_integer('num_classes', 10, 'Numbers of classes in the dataset')
flags.DEFINE_string('dataset_path', './tensorflow_datasets','Path for saving dataset.')
flags.DEFINE_string('result_path', './results','Path for saving results.')
flags.DEFINE_integer('logging_step',100,'Step number for logging')
flags.DEFINE_integer('data_buffer_size',1000,'Buffersize input pipeline.')
flags.DEFINE_integer('train_batch_size',256 ,'Batch size for training.')
flags.DEFINE_bool('cache_dataset', False, 'cache dataset (True|False).')
flags.DEFINE_string('device', 'GPU', 'Device using now(CPU|GPU)')

flags.DEFINE_integer('gen_disc_ch',64,'Number of channels in the first layer of generator and discriminator_f.')
flags.DEFINE_integer('en_ch',32,'Number of channels in the first layer of encoder.')

flags.DEFINE_float('lr_gen_en',2e-4,'Learning rate generator.')
flags.DEFINE_float('beta_1_gen_en',0.5,'Beta_1 of Generator optimizer.')
flags.DEFINE_float('beta_2_gen_en',0.999,'Beta_2 of generator optimizer.')

flags.DEFINE_float('lr_disc',2e-4,'Learning rate discriminator.')
flags.DEFINE_float('beta_1_disc',0.5,'Beta_1 of Discriminator optimizer.')
flags.DEFINE_float('beta_2_disc',0.999,'Beta_2 of discriminator optimizer.')

flags.DEFINE_integer('D_G_ratio', 1, 'Ratio of upgrading weights, discriminator VS generator & encoder')

flags.DEFINE_integer('num_epochs', 50 ,'Number of epochs to train.')

flags.DEFINE_integer('num_cont_noise', 100, 'Dimension of continous noise vector.')
flags.DEFINE_bool('conditional', True, 'Conditional or unconditional GAN')
flags.DEFINE_integer('num_emb', 10, 'Dimension of embedded label output. Only applicable when conditional')


def main(argv):
    del argv  # Unused.
    set_up_train(FLAGS)


if __name__ == '__main__':
    app.run(main)
