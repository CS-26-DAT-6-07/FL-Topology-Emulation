"""pytorchexample: A Flower / PyTorch app."""
print("---------------- DEBUG: client_app.py is working ---------------", flush=True) 
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
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
    trainloader, _ = load_partition(partition_id)

    #Load strategy_choice sent from the server side
    strategy_choice = msg.content["config"]["strategy_choice"]

    if strategy_choice == "fedavg":
        # Call the training function (for FedAvg/FedProx)
        train_loss, accuracy = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
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

        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
            "feature_vector": feature_vector,
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

        #save updated local control variate in client state for next round
        context.state["local_cv"] = ArrayRecord(new_local_cv)   

        # Construct and return reply Message
        arrays = ArrayRecord(updated_local_model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "train_acc": accuracy,
        }
        control_variate_update = ArrayRecord(cv_diff)
        metric_record = MetricRecord(metrics)
    
        content = RecordDict({
            "arrays": arrays,
            "control_variate": control_variate_update,
            "metrics": metric_record})
    
        return Message(content=content, reply_to=msg)


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
    _, valloader = load_partition(partition_id)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
        False,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
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
        features.append(output.detach().cpu())

    hook_handle = model.fc2.register_forward_hook(hook)

    with torch.no_grad():
        for batch in trainloader:
            images = batch["image"].to(device)
            model(images)

    hook_handle.remove()

    features = torch.cat(features, dim=0)   #shape: [num_images, 84]
    client_vector = features.mean(dim=0)    #shape: [84]

    print(f"Client {partition_id} final hidden layer averaged feature vector shape:", client_vector.shape)
    print(f"Client {partition_id} final hidden layer averaged feature vector:", client_vector)
    

    return client_vector.tolist()
