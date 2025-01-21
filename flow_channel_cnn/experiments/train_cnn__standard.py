"""
This experiment module implements the training of a StandardCNN model for the prediction of the laminar flow 
channel parameters $C_f$ and $S_t$ based on a dataset of flow channel geometry images.

This module extends the base experiment defined in ``train_cnn.py`` by implementing the ``construct_model`` hook 
to construct a ``StandardCNN`` model using the specified parameters. This model implements a standard CNN 
architecture whith convolutional and dense layers but without any consideration of the problem specific shift 
and flip invariance.
"""

import os
from typing import List, Union

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from flow_channel_cnn.utils import EXPERIMENTS_PATH

# == SOURCE PARAMETERS ==
# The path to the source dataset folder

# :param SOURCE_PATH:
#       The path to the source dataset folder. THis folder should contain specific .NPY files which 
#       contain the images and the labels of the dataset.
SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'dataset')
# :param NUM_VAL:
#       The number of validation samples to use for the model. This should be a number between 0 and 1.
#       If set to 0.1, 10% of the training samples will be used for validation. Could also be set to an integer
#       value to specify the number of samples to use.
NUM_VAL: Union[int, float] = 0.05

# == MODEL PARAMETERS ==
# Parameters related to the construction of the model

# :param CONV_UNITS:
#       The number of convolutional units to use for the model. This should be a list of integers where
#       each integer represents the number of filters to use for each convolutional layer. Each integer
#       in the list will represent/add one layer.
CONV_UNITS: List[int] = [16, 32, 64, 128, 256]
# :param DENSE_UNITS:
#       The number of dense units to use for the model. This should be a list of integers where each integer
#       represents the number of units to use for each dense layer. Each integer in the list will represent/add
#       one layer. The final number of units in this list should match the number of target values that 
#       the model should predict.
DENSE_UNITS: List[int] = [256, 128, 64, 2]
# :param KERNEL_SIZE:
#       The kernel size to use for the convolutional layers.
KERNEL_SIZE: int = 8
# :param USE_APS:
#       Whether to use the Adaptive Polyphase Sampling (APS) layer in the model. If set to True, the model will
#       use the APS layer after each convolutional layer to perform the strided downsampling.
USE_APS: bool = True

# == TRANING PARAMETERS ==
# Parameters related to the training process of the model

# :param EPOCHS:
#       The number of epochs to train the model for.
EPOCHS: int = 100
# :param BATCH_SIZE:
#       The batch size to use for training.
BATCH_SIZE: int = 16
# :param LEARNING_RATE:
#       The learning rate to use for training.
LEARNING_RATE: float = 1e-5
# :param DEVICE:
#       The device to use for training. This should be either 'cpu' or 'cuda'.
DEVICE: str = 'cpu'

# == EVALUATION PARAMETERS ==

# :param NUM_EXAMPLES:
#       The number of examples to use for the shift and flip invariance evaluation.
#       Selecting a high number here may incur a high runtime after the training of the model.
NUM_EXAMPLES: int = 5

__DEBUG__: bool = True

experiment = Experiment.extend(
    'train_cnn.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

from flow_channel_cnn.models import StandardCNN

@experiment.hook('construct_model', default=False, replace=True)
def construct_model(e: Experiment):
    """
    This hook is supposed to construct the model instance which can then subsequently be trained with the 
    flow channel dataset. The model that is returned needs to implement the AbstractCNN interface.
    
    ---
    
    This implementation constructs a ``StandardCNN`` model instance with the specified parameters.
    """
    e.log('Constructing StandardCNN model...')
    model = StandardCNN(
        input_dim=1,
        input_shape=(e['height'], e['width']),
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
        kernel_size=e.KERNEL_SIZE,
        stride=2,
    )
    return model

experiment.run_if_main()