# IMPORTS
import tensorflow as tf
try:
    import tf_keras as keras
    USING_TF_KERAS_PKG = True
except ImportError:
    from tensorflow import keras
    USING_TF_KERAS_PKG = False
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

# from tf_keras.models import Sequential, Model
# from tf_keras.layers import Input, Dense, BatchNormalization, Dropout
# from tf_keras.optimizers import SGD, Adam
Sequential = keras.models.Sequential
Model      = keras.models.Model
Input      = keras.layers.Input
Dense      = keras.layers.Dense
BatchNormalization = keras.layers.BatchNormalization
Dropout    = keras.layers.Dropout
SGD        = keras.optimizers.SGD
Adam       = keras.optimizers.Adam
BinaryCrossentropy = keras.losses.BinaryCrossentropy
Activation = keras.layers.Activation
import tensorflow_probability as tfp

from ray import train

# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =======================================
# DNN related function
# =======================================
# DEEP LEARNING MODEL BUILD FUNCTION
def build_DNN(input_shape, n_hidden_layers, n_hidden_units, loss, act_fun='sigmoid', optimizer:str='sgd', learning_rate=0.01, 
            use_bn=False, use_dropout=False, dropout_rate=0.5, use_custom_dropout=False, print_summary=False, 
            use_variational_layer=False, kl_weight=None, kl_use_exact=False):
    """
    Builds a Deep Neural Network (DNN) model based on the provided parameters.
    
    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    n_hidden_layers (int): Number of hidden layers in the model.
    n_hidden_units (int): Number of nodes in each hidden layer (here all hidden layers have the same shape).
    loss (keras.losses): Loss function to use in the model.
    act_fun (str, optional): Activation function to use in each layer. Default is 'sigmoid'.
    optimizer (str, optional): Optimizer to use in the model. Default is SGD.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
    use_bn (bool, optional): Whether to use Batch Normalization after each layer. Default is False.
    use_dropout (bool, optional): Whether to use Dropout after each layer. Default is False.
    use_custom_dropout (bool, optional): Whether to use a custom Dropout implementation. Default is False.
    use_variational_layer (bool, optional): Use DenseVariational layer for BMM training
    kl_weight (float, optional): scaling for KL term
    kl_use_exact (bool, optional): exact vs estimated KL computation
    
    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Setup optimizer, depending on input parameter string
    if optimizer.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    
    # ============================================
    
    # Setup a sequential model
    model = Sequential()
    
    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Add layers to the model, using the input parameters of the build_DNN function
    
    # Add first (Input) layer, requires input shape
    model.add(Input(shape=input_shape))

    DropoutLayer = myDropout if use_custom_dropout else Dropout
    
    def DenseLayer(units, activation=None, use_bias=True):
        if not use_variational_layer:
            return Dense(units, activation=activation, use_bias=use_bias)
        return tfpl.DenseVariational(
            units=units,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=kl_weight,
            kl_use_exact=kl_use_exact,
            activation=activation,
            use_bias=use_bias,
        )
    
    # Add remaining layers. These to not require the input shape since it will be infered during model compile
    for _ in range(n_hidden_layers):
        if use_bn:
            model.add(DenseLayer(n_hidden_units, activation=None, use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation(act_fun))
        else:
            model.add(DenseLayer(n_hidden_units, activation=act_fun))
            
        # Dropout is typically applied after the activation in feedforward nets
        if use_dropout:
            model.add(DropoutLayer(rate=dropout_rate))
    
    # Add final layer
    model.add(Dense(1, activation="sigmoid"))  # typical binary classifier output
    
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_DNN(config, training_config):
    '''
    Train a DNN model based on the provided configuration and data. 
    This is use in the automatic hyperparameter search and follows the format that Ray Tune expects.

    Parameters:
    config (dict): Dictionary with the configuration parameters for the model. This includes the parameters needed to build the model and can be 
                    manually set or generated by Ray Tune.
                    For convenience, the config dictionary also contains the training parameters, such as the number of epochs and batch size.
    training_config (dict): Dictionary with the training parameters, such as the number of epochs and batch size, and the data to use for training and validation (Xtrain, Ytrain, Xval, Yval).
    '''

    # A dedicated callback function is needed to allow Ray Tune to track the training process
    # This callback will be used to log the loss and accuracy of the model during training
    class TuneReporterCallback(keras.callbacks.Callback):
        """Tune Callback for Keras.
        
        The callback is invoked every epoch.
        """
        def __init__(self, logs={}):
            self.iteration = 0
            super(TuneReporterCallback, self).__init__()
    
        def on_epoch_end(self, batch, logs={}):
            self.iteration += 1
            train.report(dict(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss")))
    
    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Unpack the data tuple
    X_train, y_train, X_val, y_val = training_config["data"]  # or training_config["data_tuple"]

    # Build the model using the variables stored into the config dictionary.
    # Hint: you provide the config dictionary to the build_DNN function as a keyword argument using the ** operator.
    model = build_DNN(**config)  # ** unpacks dict into keyword args

    # Train the model (no need to save the history, as the callback will log the results).
    # Remember to add the TuneReporterCallback() to the list of callbacks.
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        verbose=0,
        callbacks=[TuneReporterCallback()],
    )

    # --------------------------------------------


# CUSTOM DROPOUT IMPLEMENTATION
# Code from https://github.com/keras-team/tf-keras/issues/81
class myDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Override training=True


# CUSTOM PRIOR AND POSTERIOR FUNCTIONS FOR THE VARIATIONAL LAYER
#  Code from https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# The prior is defined as a normal distribution with zero mean and unit variance.
# def prior(kernel_size, bias_size, dtype=None):
#     n = kernel_size + bias_size
#     prior_model = keras.Sequential(
#         [
#             tfp.layers.DistributionLambda(
#                 lambda t: tfp.distributions.MultivariateNormalDiag(
#                     loc=tf.zeros(n), scale_diag=tf.ones(n)
#                 )
#             )
#         ]
#     )
#     return prior_model


# # multivariate Gaussian distribution parametrized by a learnable parameters.
# def posterior(kernel_size, bias_size, dtype=None):
#     n = kernel_size + bias_size
#     posterior_model = keras.Sequential(
#         [
#             tfp.layers.VariableLayer(
#                 tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
#             ),
#             tfp.layers.MultivariateNormalTriL(n),
#         ]
#     )
#     return posterior_model

# Had to change the distribution type due to persistent integer overflow errors
# Apparently the MultivariateNormalTriL was creating an insanely large 
# covariance matrix
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential([
        tfpl.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
                reinterpreted_batch_ndims=1
            )
        )
    ])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n),
    ])

# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()
    
def eval_precision_recall(model, X, y_true, threshold=0.5):
    """
    Evaluate precision/recall for a binary classifier with sigmoid output.

    model: Keras model that outputs probabilities in [0, 1]
    X: features
    y_true: {0,1} labels
    threshold: probability cutoff for predicting class 1
    """
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
