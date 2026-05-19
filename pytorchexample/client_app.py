"""pytorchexample: A Flower / PyTorch app."""
#print("---------------- DEBUG: client_app.py is working ---------------", flush=True) 
import json
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn, scaffold_train
from pytorchexample.dataset.dataset import load_partition
from pytorchexample.models.xception import xception

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    model = xception()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    if partition_id <= 6:
        trainloader, _, train_seed_list, _ = load_partition(partition_id, batch_size)

    #Load strategy_choice sent from the server side
    strategy_choice = msg.content["config"]["strategy_choice"]

    if strategy_choice == "fedavg" or strategy_choice == "fedprox" or strategy_choice == "fedavgcycle":
        # Call the training function (for FedAvg/FedProx)
        train_loss, accuracy = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            seeds = json.dumps(train_seed_list)
            f.write(seeds)

        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
            "num-examples": len(trainloader.dataset),
        }
        
        model_record = ArrayRecord(model.state_dict())
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)
    elif strategy_choice == "fedtree":
        # Call the training function (for FedAvg/FedProx)
        train_loss, accuracy = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        feature_vector = extracting_clients_feature_vector(model, trainloader, device, partition_id)

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            seeds = json.dumps(train_seed_list)
            f.write(seeds)

        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
            "feature_vector": feature_vector,
            "num-examples": len(trainloader.dataset),
            "partition_id": int(context.node_config["partition-id"]),
        }
        
        model_record = ArrayRecord(model.state_dict())
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)
    
    elif strategy_choice == "scaffold":
        # Load control variate from message content
        global_control_variate = msg.content["global_cv"].to_torch_state_dict()

        # Initialize/load client control variate
        if "local_cv" in context.state:
            local_control_variate = context.state["local_cv"].to_torch_state_dict()
        else:
            local_control_variate = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    
        # Call the scaffold training function
        train_loss, accuracy, updated_local_model, new_local_cv, cv_diff = scaffold_train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            global_control_variate,
            local_control_variate
        )

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            seeds = json.dumps(train_seed_list)
            f.write(seeds)

        #save updated local control variate in client state for next round
        context.state["local_cv"] = ArrayRecord(new_local_cv)   

        # Construct and return reply Message
        arrays = ArrayRecord(updated_local_model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
            "num-examples": len(trainloader.dataset),
        }
        control_variate_update = ArrayRecord(cv_diff)
        metric_record = MetricRecord(metrics)
    
        content = RecordDict({
            "arrays": arrays,
            "control_variate": control_variate_update,
            "metrics": metric_record})
    
        return Message(content=content, reply_to=msg)
    else:
        raise Exception("Did not give proper strategy in toml file")
    


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = xception()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader, _, val_seed_list = load_partition(partition_id, batch_size)

    #Load strategy_choice sent from the server side (needed for experiment folder naming)
    strategy_choice = msg.content["config"]["strategy_choice"]

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
        False,
    )

    with open(f"experiment_{strategy_choice}/client_{partition_id}_eval_seeds.json", "w") as f:
        seeds = json.dumps(val_seed_list)
        f.write(seeds)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

def extracting_clients_feature_vector(model, trainloader, device, partition_id):
    model.eval()
    features = []

    def hook(module, input, output):
        #print(f"\nClient {partition_id} activation shape:", output.shape)
        #print(f"Client {partition_id} activations:", output)

        #Deatach and squash the spatial dimensions immediately
        #output is [batch, 2048, H, W]. dim=(2,3) removes H and W.
        pooled_output = torch.mean(output, dim=(2, 3)).detach().cpu() 
        features.append(pooled_output)

        hook_handle = model.bn4.register_forward_hook(hook) #Choose bn4 instead of last hidden layer as we want general features 

        max_batches = 5 
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if i >= max_batches:
                    break
                images = batch["image"].to(device)
                model(images)
                del images #Clears the GPU memory immediately, keep for now

        hook_handle.remove()

        all_features = torch.cat(features, dim=0)  #shape: [Total_Images, 2048]
        client_vector = all_features.mean(dim=0)   #shape: [2048]

        #print(f"Client {partition_id} final hidden layer averaged feature vector shape:", client_vector.shape)
        #print(f"Client {partition_id} final hidden layer averaged feature vector:", client_vector)

        return client_vector.tolist()
