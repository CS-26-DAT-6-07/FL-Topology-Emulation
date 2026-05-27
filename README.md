---
dataset: [Fed-ISIC19]
framework: [flower, torch, torchvision]
---

# Mitigating Attribute Skew Using Network Topology Emulation in Federated Learning Systems

This paper proposes FedTree and FedAvgCycle as an emulation of hierarchical and cyclic topology in an attempt to mitigate attribute skew using the dataset Fed-ISIC19.

## Set up the project

Install Flower:

```shell
pip install flwr
```

Install dependencies in `pyproject.toml`:

```shell
pip install -e .
```

## Run the project

```bash
# Run with the default federation
flwr run . --stream
```

### Configuration
Configuration file is located in `pyproject.toml`, where the configs can be changed, including the strategy used:
- `fedavg`
- `fedprox`
- `scaffold`
- `fedtree`
- `fedavgcycle`

You can edit the configuration file or override it with a bash command

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

