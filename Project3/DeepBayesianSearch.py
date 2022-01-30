
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.model_params.Deep_architectures import Models_Constructor



def Train_Test_keras():


    PROJECT_PATH="/content/drive/MyDrive/Data_Science_Competitions/Project3/"
    train_df=pd.read_csv(PROJECT_PATH+"Data/train.csv")
    
    features_df = train_df.drop(columns=['label'])
    targets_df = train_df['label']

    num_classes = len(targets_df.value_counts().index)
    input_shape = (28, 28, 1)

    X_train, X_test, y_train, y_test = train_test_split(features_df, targets_df, train_size=0.8)
   
    X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
        )
    ]

    Builders=Models_Constructor(num_classes=num_classes)
    bayesian_tuner=kt.BayesianOptimization(Builders.ConvNet_builder,
                                       objective='val_accuracy',
                                       max_trials=40,
                                       executions_per_trial=3,
                                       overwrite=False,
                                       beta=3.6,
                                       directory="My_Dir",
                                       project_name="DigitRecognizer")

    bayesian_tuner.search(X_train, y_train, epochs=100, callbacks=train_callbacks, validation_split=0.3)

    best_model=bayesian_tuner.get_best_models()[0]
    y_pred = best_model.predict(X_test)
    y_pred=np.argmax(y_pred, axis=1, out=None)
    y_test=np.argmax(y_test, axis=1, out=None)

    from sklearn.metrics import confusion_matrix

    plt.rcParams['figure.figsize'] = [20, 15]
    plt.title("Confusion Matrix Best Score")
    conf_matrix=confusion_matrix(y_test, y_pred)
    #sn.heatmap(conf_matrix, annot=True, cmap="inferno")
    sn.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, fmt='.3%', cmap="inferno")
    plt.savefig("Confusion_Matrix.png")




    


    







    

