"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.custom_strategy import TreeStrategy, Scaffold

from pytorchexample.task import Net, test
from pytorchexample.models.xception import xception
from pytorchexample.dataset.dataset import load_centralized_dataset, init_dataset

init_dataset(seed=42,rep=0)

# Create ServerApp
app = ServerApp()

#Hardcoded two edge servers and which clients belong to each edge server
EDGE_GROUPS = {
    0: [0, 1, 2],
    1: [3, 4, 5],
}

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    # Load global model
    global_model = xception()
    arrays = ArrayRecord(global_model.state_dict())

    #Initialize strategy
    """
    strategy = TreeStrategy(
        edge_groups=EDGE_GROUPS,
        fraction_evaluate=fraction_evaluate,
    )
    """
    strategy = Scaffold(
        initial_parameters=arrays,
        lr=lr,
        fraction_evaluate=fraction_evaluate,
    )
    
    #strategy = FedAvg(
        #fraction_train=0.5,#fraction of nodes to involve in a round of training
    #    fraction_evaluate=fraction_evaluate,
        #min_available_nodes=100, #minimum connected nodes required before FL starts
    #    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

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
    test_loss, test_acc = test(model, test_dataloader, device)

    #Should print results correctly
    #print(f"Round {server_round} - Accuracy: {test_acc}, Loss: {test_loss}")

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
