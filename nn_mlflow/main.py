import data_preparation
import feature_engineering
import model_training
import prediction

def main():

    print('Starting')
    splitted_data = data_preparation.prepare_data()

    print('Featuring')
    features = feature_engineering.prepare_data(splitted_data)

    print('Modeling')
    model = model_training.train_model(features[0], splitted_data[1], features[1], splitted_data[2] )

    print('Predicting')
    #Show results
    prediction = prediction.prediction(model, features['X_test'], features['y_test'], features['y_pred'])

if __name__ == "__main__":
    main()