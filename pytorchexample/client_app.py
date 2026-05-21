"""pytorchexample: A Flower / PyTorch app."""
#print("---------------- DEBUG: client_app.py is working ---------------", flush=True) 
import json
import os
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
    model = xception()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]

    if partition_id <= 6:
        trainloader, _, train_seed_list, _ = load_partition(partition_id, batch_size)
    else:
        print(f"Client Nr. {partition_id} EXITING!!!")
        exit(555)

    strategy_choice = msg.content["config"]["strategy_choice"]
    
    # FIX 3: Ensure the experiment directory actually exists before writing to it
    os.makedirs(f"experiment_{strategy_choice}", exist_ok=True)

    if strategy_choice in ["fedavg", "fedprox", "fedavgcycle"]:
        train_loss, accuracy = train_fn(model, trainloader, context.run_config["local-epochs"], msg.content["config"]["lr"], device)

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            f.write(json.dumps([seed for seed in train_seed_list]))

        metrics = {"train_loss": train_loss, "train_acc": accuracy, "num-examples": len(trainloader.dataset)}
        content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    elif strategy_choice == "fedtree":
        #feature_vector = extracting_clients_feature_vector(model, trainloader, device, partition_id)
        feature_container = {"data": []}
        
        def hook(module, input, output):
            if len(feature_container["data"]) < 5:
                tensor = torch.nn.functional.relu(output)
                pooled = torch.mean(tensor, dim=(2, 3)).detach().cpu()
                feature_container["data"].append(pooled)

        hook_handle = model.bn4.register_forward_hook(hook)

        # Train locally
        train_loss, accuracy = train_fn(model, trainloader, context.run_config["local-epochs"], msg.content["config"]["lr"], device)

        hook_handle.remove()

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            f.write(json.dumps([seed for seed in train_seed_list]))

        feature_vector = [float(x) for x in torch.cat(feature_container["data"]).mean(dim=0).tolist()] if feature_container["data"] else [0.0]*2048

        metrics = {
            "train_loss": train_loss, 
            "train_acc": accuracy, 
            "num-examples": len(trainloader.dataset),
            "partition_id": partition_id,
            "feature_vector": feature_vector
        }
        content = RecordDict()
        content.arrays["arrays"] = ArrayRecord(model.state_dict())
        content.metrics["metrics"] = MetricRecord(metrics)
        return Message(content=content, reply_to=msg)
    
    elif strategy_choice == "scaffold":
        global_control_variate = msg.content["global_cv"].to_torch_state_dict()

        if "local_cv" in context.state:
            local_control_variate = context.state["local_cv"].to_torch_state_dict()
        else:
            local_control_variate = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    
        train_loss, accuracy, updated_local_model, new_local_cv, cv_diff = scaffold_train(
            model, trainloader, context.run_config["local-epochs"], msg.content["config"]["lr"], device, global_control_variate, local_control_variate
        )

        with open(f"experiment_{strategy_choice}/client_{partition_id}_train_seeds.json", "w") as f:
            f.write(json.dumps([seed for seed in train_seed_list]))

        context.state["local_cv"] = ArrayRecord(new_local_cv)   

        metrics = {"train_loss": train_loss, "train_acc": accuracy, "num-examples": len(trainloader.dataset)}
        
        # FIX 2: Safely nest BOTH ArrayRecords under the strictly allowed `.arrays` namespace
        content = RecordDict()
        content.arrays["arrays"] = ArrayRecord(updated_local_model.state_dict())
        content.arrays["control_variate"] = ArrayRecord(cv_diff)
        content.metrics["metrics"] = MetricRecord(metrics)
    
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
        nonproxy_val_seed_list = []
        for seed in val_seed_list:
            nonproxy_val_seed_list.append(seed)
        seeds = json.dumps(nonproxy_val_seed_list)
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

    """
    def hook(module, input, output): 
        tensor = output[0] if isinstance(output, tuple) else output
        
        #Apply ReLU to get post-activation features
        tensor = torch.nn.functional.relu(tensor)
        
        # Global Average Pooling
        pooled_output = torch.mean(tensor, dim=(2, 3)).detach().cpu()
        feature_container["data"].append(pooled_output)

    print(f"DEBUG: Registering hook on model.bn4 for partition {partition_id}")
    hook_handle = model.bn4.register_forward_hook(hook)

    max_batches = 5 
    batches_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            if i >= max_batches:
                break
            images = batch["image"].to(device).float()
            model(images) 
            batches_processed += 1
            del images 

    hook_handle.remove()

    if not feature_container["data"]:
        print(f"WARNING: feature_container is empty for partition {partition_id}!")
        return [0.0] * 2048 

    all_features = torch.cat(feature_container["data"], dim=0)
    client_vector = all_features.mean(dim=0)

    print(f"Client {partition_id} final hidden layer averaged feature vector shape:", client_vector.shape)
    print(f"Client {partition_id} final hidden layer averaged feature vector:", client_vector)
    
    return [float(x) for x in client_vector.tolist()]
    """