import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import mlflow
import time
import os

def execute(event):

    # Using os.getenv() is best, because it returns None if the environment variable is not set, instead of crashing.
    mlflow_experiment = event["experiment_name"]
    mlflow_master_ip = event["master_ip"]

    mlflow.set_tracking_uri(mlflow_master_ip)
    try:
        experiment_id = mlflow.get_experiment_by_name(mlflow_experiment).experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(mlflow_experiment)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        
        print("Parsing event dictionary")
        layers = event["layers"]
        epochs = event["epochs"]
        batch_size = event["batch_size"]

        print("Started")
        x_train = np.random.random((1000, 20))
        y_train = np.random.randint(2, size=(1000, 1))
        x_test = np.random.random((100, 20))
        y_test = np.random.randint(2, size=(100, 1))

        # Build a model
        model = Sequential()
        for i in layers:
            model.add(parse_layer(i))

        # Configure the learner
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        # Train
        print("Training beginning")
        model.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size)
        print("Done")
        # Evaluate		  
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        count = 0
        for lay in layers:
            count += 1
            if len(lay) >= 3:
                mlflow.log_param(str(count) + "_" + lay[0],"Layer {0} com profundidade {1} e ativação {2}".format(lay[0], lay[1], lay[-1]))
            else: 
                mlflow.log_param(str(count) + "_" + lay[0],"Layer {0} com profundidade {1}".format(lay[0], lay[1]))
        mlflow.log_metric("ScoreX", score[0])
        mlflow.log_metric("ScoreY", score[1])

    return {
        'statusCode': 200,
        'body': json.dumps("Finished"),
        'event': json.dumps(event)
    }


# (Layer Type, depth, activation, input_dim)
def parse_layer(lay):
    if lay[0] == "Dropout":
        return Dropout(lay[1])
    if lay[0] == "Dense":
        if len(lay) == 4:
            return Dense(lay[1], activation=lay[2], input_dim=lay[3])
        return Dense(lay[1], activation=lay[2])
    ## Can add more layer options

def ker_lay_str(lType: str, depth: int,  input_dim: int = None, activation: str = "relu"):
    if lType == "Dropout":
        return [lType, depth]
    if input_dim is not None:
        return [lType, depth, activation, input_dim]
    return [lType, depth, activation]

if __name__ == "__main__":
    payload = {"layers":os.getenv("layers"),
                "epochs":os.getenv("epochs"),
                "batch_size":os.getenv("batch_size"),
                "experiment_name":os.getenv("experiment_name"),
                "master_ip":os.getenv("master_ip")}
    # start = time.perf_counter()
    lambda_responses = execute(payload)
    # print(lambda_responses)
    # print(f"Tempo total de execução: {time.perf_counter()-start}\n")