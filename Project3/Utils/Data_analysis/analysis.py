import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def plot_digits(df):
    '''Plot the possible Digits of a DataFrame
    Args:
        df, DataFrame
    Return:
        fig, figure with one example of each digit
    '''
    plt.rcParams["figure.figsize"] = (14,14)
    dim_plot=int(np.sqrt(len(df['label'].value_counts().index)))+1
    fig = plt.figure()

    for index, digit in enumerate(df['label'].value_counts().index):

        ax=fig.add_subplot(dim_plot, dim_plot, index+1)
        ax.set_title('Digit {}'.format(digit))
        plt.imshow(df[df['label']==digit].drop(columns='label').iloc[0].values.reshape(28, 28), cmap='inferno')
        plt.colorbar()
        
    plt.savefig("/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/img/Digits/Digitis_example.png")

    return fig

def hist_of_digits(df):
    '''Plot the histogram of the digits
    Args:
        df, DataFrame
    Return:
        None
    '''
    label_hist=''
    for digit, count in df['label'].value_counts().items():
        label_hist+='{} : {} \n'.format(digit, count)

    df['label'].hist(label=label_hist)
    plt.title("Histogram")
    plt.legend()

    plt.savefig("/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/img/Digits/Digitis_hist.png")


def drop_no_information(df):
    '''Drop columns with no information in a DataFrame
    Args:
        df, DataFrame
    Return:
        df[columns_to_keep], new DataFrame with relevant information
        columns_to_keep, columns with relevant information
    '''
    columns_to_keep=[column for column, boolean in (df.drop(columns='label').std()!=0).items() if boolean]
    
    sn.heatmap((df.drop(columns='label').std()==0).values.reshape(28, 28))
    plt.title("Pixels with no Information")
    plt.savefig("/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/img/Digits/Pixels_no_inf.png")

    return df[columns_to_keep], columns_to_keep