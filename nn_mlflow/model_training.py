# import os
# import random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras_tuner.tuners import RandomSearch

# def reset_seeds():
#     os.environ['PYTHONHASHSEED'] = str(2)
#     tf.random.set_seed(2)
#     np.random.seed(2)
#     random.seed(2)

# def train_model(X_train, y_train, X_val, y_val):   
    
#     def build_model(hp):
#         model = keras.Sequential([
#             keras.Input(shape=(X_train.shape[1],), name="input"),
#             layers.BatchNormalization(),
#             layers.Dense(hp.Int('units1', 16, 64, step=16), activation=keras.activations.selu, name="hidden_layer1", kernel_initializer=keras.initializers.LecunNormal),
#             keras.layers.Dropout(rate=hp.Float('dropout2', 0, 0.5, step=0.1)),
#             layers.BatchNormalization(),
#             layers.Dense(hp.Int('units2', 8, 32, step=8), activation=keras.activations.selu, name="hidden_layer2", kernel_initializer=keras.initializers.LecunNormal),
#             layers.Dense(1, activation="sigmoid", name="output")
#         ])

#         model.compile(
#             optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
#             loss=tf.keras.losses.BinaryCrossentropy(),
#             metrics=['accuracy']
#         )

#         return model
        
#     tuner = RandomSearch(
#         build_model,
#         objective='val_accuracy',
#         max_trials=10,
#         executions_per_trial=3,
#         directory='tuner_dir',
#         project_name='my_model'
#     )

#     tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

#     best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

#     model = keras.Sequential(
#         [
#             keras.Input(shape=(X_train.shape[1],), name="input"),
#             layers.BatchNormalization(),
#             layers.Dense(8, activation=keras.activations.selu, name="hidden_layer2", kernel_initializer=keras.initializers.LecunNormal, kernel_regularizer='l1_l2'),
#             layers.Dense(1, activation="sigmoid", name="output")
#         ]
#     )

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=100, decay_rate=0.96, staircase=True)),
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=['accuracy']
#     )

#     cb = tf.keras.callbacks.EarlyStopping(
#         patience=5,
#         restore_best_weights=True
#     )

#     reset_seeds()

#     fitted_model = model.fit(
#         x=X_train,
#         y=y_train,
#         batch_size=32,
#         epochs=50,
#         validation_data=(X_val, y_val),
#         callbacks=cb, # use early stopping
#     )
    
#     return fitted_model, model

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_tuner.tuners import RandomSearch
import mlflow

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)

with mlflow.start_run():
    mlflow.log_param("model_training", "model_training.py")
    mlflow.log_metric("model_training", 1)
    def train_model(X_train, y_train, X_val, y_val):   
        
        def build_model(hp):
            model = keras.Sequential([
                keras.Input(shape=(X_train.shape[1],), name="input"),
                layers.BatchNormalization(),
                layers.Dense(hp.Int('units1', 16, 64, step=16), activation=keras.activations.selu, name="hidden_layer1", kernel_initializer=keras.initializers.LecunNormal),
                keras.layers.Dropout(rate=hp.Float('dropout2', 0, 0.5, step=0.1)),
                layers.BatchNormalization(),
                layers.Dense(hp.Int('units2', 8, 32, step=8), activation=keras.activations.selu, name="hidden_layer2", kernel_initializer=keras.initializers.LecunNormal),
                layers.Dense(1, activation="sigmoid", name="output")
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )

            return model

        tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=1,
            directory='tuner_dir',
            project_name='my_model'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

        # Retrieve best hyperparameters and fit the final model
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

        # Log the best hyperparameters to MLflow
        for key, value in best_hyperparameters.values.items():
            mlflow.log_param(key, value)

        # Fit the final model with the best hyperparameters
        model = tuner.hypermodel.build(best_hyperparameters)
        cb = tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
        reset_seeds()
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[cb],
        )

        # Log the training history to MLflow
        for metric_name in history.history.keys():
            metric_values = history.history[metric_name]
            for i in range(len(metric_values)):
                mlflow.log_metric(metric_name, metric_values[i], step=i)

        return history, model
