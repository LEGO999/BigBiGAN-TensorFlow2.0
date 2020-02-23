import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


def get_fixed_random(config, num_to_generate=100):
    seed_cont = tf.random.truncated_normal([num_to_generate, 100])
    # number of the generated images and numbers of noise per samples.
    seed_cat = tf.math.mod(tf.range(0, num_to_generate), config.num_classes)
    return seed_cont, seed_cat


def generate_images(generator, z_input, c_input, config):
    if not config.conditional:
        c_input = None
    predictions = generator(z_input, c_input, training=False)
    fig = plt.figure(figsize=(10, 10))
    if config.dataset == ('mnist' or 'fashion_mnist'):  # color channel of the dataset
        for i in range(predictions.shape[0]):
            plt.subplot(10, 10, i + 1)  # 10*10 subplots

            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
    else:
        for i in range(predictions.shape[0]):
            plt.subplot(10, 10, i + 1)  # 10*10 subplots

            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')

    return _fig2data(fig)


def _fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3d numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    canvas = FigureCanvas(fig)
    # The canvas the figure renders into. Calls the draw and print fig methods, creates the renderers
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    # get_size_inches(): Returns the current size of the figure in inches (1in == 2.54cm) as an numpy array.
    # fig.get_dpi: Return the resolution in dots per inch as a float.
    image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(height.astype(np.int32), width.astype(np.int32), 3)
    # canvas.tostring_rgb() Get the image as an RGB byte string. And reshape them into our width and height.
    return image
