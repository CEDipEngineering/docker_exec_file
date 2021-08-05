import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import mlflow
import time
import os

def execute(event):
    start = time.time() # Store UNIX timestamp at start
    # Using os.getenv() is best, because it returns None if the environment variable is not set, instead of crashing.
    mlflow_experiment = event["experiment_name"]
    if(event["master_ip"] is not None):
        mlflow_master_ip = event["master_ip"]
        mlflow.set_tracking_uri(mlflow_master_ip)
    
    try:
        experiment_id = mlflow.get_experiment_by_name(mlflow_experiment).experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(mlflow_experiment)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        
        # print("Parsing event dictionary")
        layers = event["layers"]
        epochs = event["epochs"]
        batch_size = event["batch_size"]

        # print("Started")
        x_train = np.random.random((1000, 20))
        y_train = np.random.randint(2, size=(1000, 1))
        x_test = np.random.random((100, 20))
        y_test = np.random.randint(2, size=(100, 1))

        # Build a model
        model = Sequential()
        for i in range(len(layers)):
            model.add(parse_layer(layers[i]))

        # Configure the learner
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        # Train
        # print("Training beginning")
        model.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size)
        # print("Done")
        # Evaluate		  
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        
        # print("Logging metrics...")
        count = 0
        for lay in layers:
            count += 1
            if len(lay) >= 3:
                mlflow.log_param(str(count) + "_" + lay[0],"Layer {0} com profundidade {1} e ativação {2}".format(lay[0], lay[1], lay[-1]))
            else: 
                mlflow.log_param(str(count) + "_" + lay[0],"Layer {0} com profundidade {1}".format(lay[0], lay[1]))
        mlflow.log_metric("ScoreX", score[0])
        mlflow.log_metric("ScoreY", score[1])
        mlflow.log_metric("TimeElapsed(s)", time.time()-start)
        # print("Logging complete!")

    return {
        'statusCode': 200,
        'body': json.dumps("Finished"),
        'event': json.dumps(event)
    }


# (Layer Type, depth, activation, input_dim)
def parse_layer(lay):
    if lay[0] == "Dropout":
        return Dropout(float(lay[1]))
    if lay[0] == "Dense":
        if len(lay) == 4:
            return Dense(int(lay[1]), activation=lay[2], input_dim=int(lay[3]))
        return Dense(int(lay[1]), activation=lay[2])
    ## Can add more layer options

def ker_lay_str(lType: str, depth: int,  input_dim: int = None, activation: str = "relu"):
    if lType == "Dropout":
        return [lType, depth]
    if input_dim is not None:
        return [lType, depth, activation, input_dim]
    return [lType, depth, activation]

if __name__ == "__main__":
    

    # For testing use of environment variables
    os.environ["layersTypes"] = "Dense Dropout Dense Dropout Dense"
    os.environ["layersSizes"] = "64 0.5 64 0.5 1"
    os.environ["layersActivations"] = "relu None relu None sigmoid"
    os.environ["epochs"] = "3"
    os.environ["MLFLOW_TRACKING_URI"] = "http://3.143.228.222:5000"
    os.environ["batch_size"] = "128"
    os.environ["experiment_name"] = "Random Training"
    os.environ["master_ip"] = "http://3.143.228.222:5000"
    
    
    layers = [(t,s,k) for t,s,k in zip(os.getenv("layersTypes").split(), os.getenv("layersSizes").split(), os.getenv("layersActivations").split())]
    # Corrige primeira layer pra indicar input_dim
    layers[0] = (layers[0][0],layers[0][1], layers[0][2], 20)
    payload = {"layers":layers,
                "epochs":int(os.getenv("epochs")),
                "batch_size":int(os.getenv("batch_size")),
                "experiment_name":os.getenv("experiment_name").strip(),
                "master_ip":os.getenv("master_ip").strip()}
    # print(f"Payload: {payload}\n")
    # start = time.perf_counter()
    lambda_responses = execute(payload)
    # print(lambda_responses)
    # print(f"Tempo total de execução: {time.perf_counter()-start}\n")
    