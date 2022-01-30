def NeighborComparative(df):

    numeric_features = df.drop(
        columns='Id')._get_numeric_data().columns.tolist()

    for feature in numeric_features:

        neighbors_dict = {}
        for neighbor in df['Neighborhood'].value_counts().index:

            neighbors = df[df['Neighborhood'] == neighbor]
            neighbors_dict[neighbor] = sum(neighbors[feature])

        for index in range(df.shape[0]):
            df.loc[index:index+1, str(feature)+'Comp'] = df[index:index+1][feature].values[0]/(
                neighbors_dict[df[index:index+1]['Neighborhood'].values[0]])

    return df
