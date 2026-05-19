"""pytorchexample: A Flower / PyTorch app."""


#import torch and models dataset task before flwr stuff, otherwise crash on some systems
import torch
import json

from pytorchexample.task import test 
from pytorchexample.models.xception import xception
from pytorchexample.dataset.dataset import load_centralized_dataset, init_dataset

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from pytorchexample.custom_strategy import TreeStrategy, Scaffold, FedAvgCyclic

init_dataset(seed=42,rep=0)

# Create ServerApp
app = ServerApp()

EDGE_GROUPS = {
    0: [0, 1, 2],
    1: [3, 4, 5],
}

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    strategy_choice: str = context.run_config["strategy-choice"]
    mu_prox: float = context.run_config["mu-prox"] #for fedprox only
    
    # Load global model
    global_model = xception()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = None

    #Initialize strategy
    if strategy_choice == "fedavg":
        strategy = FedAvg(
            fraction_train=fraction_train,#fraction of nodes to involve in a round of training
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=6, #minimum connected nodes required before FL starts
        )
    elif strategy_choice == "fedtree":
        strategy = TreeStrategy(
            edge_groups=EDGE_GROUPS,
            fraction_evaluate=fraction_evaluate,
        )
    elif strategy_choice == "scaffold":
        strategy = Scaffold(
            initial_parameters=arrays,
            lr=lr,
            fraction_evaluate=fraction_evaluate,
        )
    elif strategy_choice == "fedprox":
        strategy = FedProx(
            fraction_train=fraction_train,#fraction of nodes to involve in a round of training
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=6, #minimum connected nodes required before FL starts
            mu=mu_prox, #proximal term constant
         )
    elif strategy_choice == "fedavgcycle":
        strategy = FedAvgCyclic(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=6,
        )
    else:
        raise Exception("No Strategy chosen in the toml file / run_config")
    
    
    

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "strategy_choice": strategy_choice}),
        evaluate_config=ConfigRecord({"strategy_choice": strategy_choice}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    #Saving final metrics to disk
    print("\nSaving final metrics to disk...")
    with open(f"experiment_{strategy_choice}/train_metrics_clientapp", "w") as f:
        metrics = json.dumps(result.train_metrics_clientapp)
        f.write(metrics)
    with open(f"experiment_{strategy_choice}/evaluate_metrics_clientapp", "w") as f:
        metrics = json.dumps(result.evaluate_metrics_clientapp)
        f.write(metrics)
    with open(f"experiment_{strategy_choice}/evaluate_metrics_serverapp", "w") as f:
        metrics = json.dumps(result.evaluate_metrics_serverapp)
        f.write(metrics)


    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = xception()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc, precision, recall, f1, support = test(
        model, test_dataloader, device, global_eval=True
    )

    metrics = {
        "global_accuracy":  test_acc,
        "global_loss":      test_loss,
        "global_precision": precision,
        "global_recall":    recall,
        "global_f1":        f1,
    }
    for i, s in enumerate(support):
        metrics[f"global_support_class_{i}"] = float(s)

    return MetricRecord(metrics)
