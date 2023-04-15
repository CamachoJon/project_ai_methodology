from sklearn.metrics import classification_report
import tensorflow as tf
import mlflow

with mlflow.start_run():
    mlflow.log_param("prediction", "prediction.py")
    mlflow.log_metric("prediction", 1)
    def prediction(model, X_test, y_test):
        y_pred = (model.predict(X_test) > 0.5) * 1
        print(classification_report(y_test, y_pred))

        # General metrics 
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = round(precision_score(y_test, y_pred.astype(int), average='weighted'), 3)
        recall = round(recall_score(y_test, y_pred.astype(int), average='weighted'), 3)
        f1 = round(f1_score(y_test, y_pred.astype(int), average='weighted'), 3)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1 score: ', f1)
        # Log the metrics to MLflow
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 score", f1)
        
        # Save model
        tf.saved_model.save(model, 'saved_model')