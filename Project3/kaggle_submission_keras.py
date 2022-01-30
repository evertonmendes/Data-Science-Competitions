
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.model_params.Deep_architectures import Models_Constructor


def create_kaggle():

    test_df = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/test.csv')
    
    submission_df = pd.DataFrame(data=[index+1 for index in test_df.index], columns=['ImageId'])
    
    kaggle_test=test_df.values
    kaggle_test=kaggle_test.astype("float32") / 255
    kaggle_test = kaggle_test.reshape(kaggle_test.shape[0], 28, 28, 1)

    bayesian_tuner=kt.BayesianOptimization(Builders.ConvNet_builder,
                                       objective='val_accuracy',
                                       max_trials=80,
                                       executions_per_trial=2,
                                       overwrite=False,
                                       beta=3.6,
                                       directory="/content/drive/MyDrive/Data_Science_Competitions/Project3/My_Dir",
                                       project_name="DigitRecognizer")

    best_model=bayesian_tuner.get_best_models()[0]
    kaggle_pred = best_model.predict(kaggle_test)

    submission_df = pd.DataFrame(data=[index+1 for index in test_df.index], columns=['ImageId'])
    submission_df['Label'] = np.argmax(kaggle_pred, axis=1, out=None)
    submission_df.to_csv('kaggle_submission.csv', index=False)
