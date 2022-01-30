#from keras_self_attention import seq_self_attention
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten, TimeDistributed, Concatenate, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
#from tensorflow.python.keras.layers.dense_attention import Attention
#from attention import Attention
from tensorflow.keras.models import Model
#from keras_self_attention import SeqSelfAttention


class Models_Constructor():

    def __init__(self, num_classes, input_shape = (28, 28, 1)):

        self.num_classes=num_classes
        self.input_shape=input_shape


    def ConvNet_builder(self, hp):

        # defining a set of hyperparametrs for tuning
        conv_filters=hp.Int(name = 'filters', min_value = 1, max_value = 256, step = 1)
        conv_kernel=hp.Int(name = 'kernel_size', min_value = 4, max_value = 8, step = 1)
        conv_pool_size=hp.Int(name = 'pool_size', min_value = 1, max_value = 2, step = 1)
        conv_activation = hp.Choice(name='activation', values = ['tanh', 'relu','sigmoid', 'softmax', 'softplus'], ordered = False)
        #dropout layer
        dropout = hp.Float(name = 'dropout', min_value=0, max_value=.5, step=0.05)
        #Dense layers
        activation = hp.Choice(name='activation', values = ['tanh', 'relu','sigmoid', 'softmax', 'softplus'], ordered = False)
        activation1 = hp.Choice(name='activation1', values = ['tanh', 'relu','sigmoid', 'softmax', 'softplus'], ordered = False)
        activation2 = hp.Choice(name='activation2', values = ['tanh', 'relu','sigmoid', 'softmax', 'softplus'], ordered = False)
        units=hp.Int(name = 'units', min_value = 1, max_value = 256, step = 1)
        #learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        optimizer=hp.Choice('optimizer', ['sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'ftrl'])
        
        #model
        model=Sequential()
        model.add(Conv2D(2*conv_filters, kernel_size=(conv_kernel, conv_kernel), activation=conv_activation, input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(conv_pool_size, conv_pool_size)))
        model.add(Conv2D(conv_filters, kernel_size=(conv_kernel, conv_kernel), activation=conv_activation))
        model.add(MaxPooling2D(pool_size=(conv_pool_size, conv_pool_size)))
        model.add(Flatten())
        model.add(Dropout(dropout))
        model.add(Dense(units=units, activation=activation1))
        model.add(Dense(self.num_classes, activation=activation2))

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

      



