import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == '__main__':

    #Read the Data
    df=pd.read_csv('corona_data.csv')
    train, test = data_split(df, 0.2)
    x_train = train[['Fever', 'BodyPains', 'Age', 'RunnyNose', 'Difficulty_in_Breath']].to_numpy()
    x_test = test[['Fever', 'BodyPains', 'Age', 'RunnyNose', 'Difficulty_in_Breath']].to_numpy()
    y_train = train[['infection_Probability']].to_numpy().reshape(5684 , )
    y_test = test[['infection_Probability']].to_numpy().reshape(1420 , )

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()