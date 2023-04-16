import data_preparation
import feature_engineering
import model_training
import prediction
import mlflow

with mlflow.start_run():
    mlflow.log_param("main", "main.py")
    def main():

        print('Starting')
        splitted_data = data_preparation.prepare_data()
        
        print('Featuring')
        features = feature_engineering.prepare_data(splitted_data)

        print('Modeling')
        #train model
        y_train = features[0][:, -1]
        y_val = features[1][:, -1]
        history, model = model_training.train_model(features[0], y_train, features[1], y_val)

        print('Predicting')
        #Show results
        predictions = prediction.prediction(model, features[2], features[3])

if __name__ == "__main__":
    main()
    with mlflow.start_run():
        data_preparation.run()
        feature_engineering.run()
        model_training.run()
        prediction.run()