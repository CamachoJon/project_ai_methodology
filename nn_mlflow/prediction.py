from sklearn.metrics import classification_report
import tensorflow as tf

def prediction(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5) * 1
    print(classification_report(y_test, y_pred))

    # General metrics 
    from sklearn.metrics import precision_score, recall_score, f1_score
    print('Precision: ', round(precision_score(y_test, y_pred.astype(int), average='weighted'), 3))
    print('Recall: ', round(recall_score(y_test, y_pred.astype(int), average='weighted'), 3))
    print('F1 score: ', round(f1_score(y_test, y_pred.astype(int), average='weighted'), 3))

    # Save model
    tf.saved_model.save(model, 'saved_model')