from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import shap
import mlflow
import shap_visualizations

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
        
        print('Starting SHAP')
        # Compute Shapley values for your input data using the TreeExplainer object
        explainer = shap.Explainer(model, X_test)

        print('Explainer worked')
        shap_values_point = explainer(X_test[:100])
        shap_values_dataset = explainer(X_test)

        # Print the Shapley values and corresponding feature names
        print(shap_values_point)
        print(shap_values_dataset)

        print(explainer.feature_names)

        # Create a SHAP force plot for a single data point
        shap_visualizations.create_force_plot(
            model=model,
            x=X_test[0],
            output_file='force_plot.png'
        )
        
        # Create a SHAP summary plot for all data points
        shap_visualizations.create_summary_plot(
            model=model,
            X=X_test,
            output_file='summary_plot.png'
        )

        # Create a SHAP summary plot for each class
        shap_visualizations.create_class_summary_plot(
            model=model,
            X=X_test,
            y=y_test,
            output_file='class_summary_plot_{1}.png'
        )

        # Save model
        tf.saved_model.save(model, 'saved_model')
