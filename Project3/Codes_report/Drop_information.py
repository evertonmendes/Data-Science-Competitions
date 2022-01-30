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