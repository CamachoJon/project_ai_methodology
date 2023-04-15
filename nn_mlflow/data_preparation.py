import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data():
    # Load the dataset
    df = pd.read_csv('../Dataset/projects.csv')
    df_original = df.copy()
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into training, validation, and test sets
    X = df.drop(columns=['state', 'name'])
    y = df['state']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=42, test_size=0.1, stratify=y_train_val)

    # Return all of the variables in a list
    return [X_train, y_train, X_val, y_val, X_test, y_test]

splitted_data = prepare_data()
print('Shape of splitted_data', splitted_data[0].shape)