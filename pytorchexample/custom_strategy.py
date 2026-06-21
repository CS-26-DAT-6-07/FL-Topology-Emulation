import math                         
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Iterable, List, Tuple, Optional
import numpy as np
import torch 

from collections.abc import Iterable
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict, MessageType, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns, log
from flwr.server.strategy.aggregate import aggregate
from flwr.serverapp.strategy.strategy_utils import sample_nodes


class TreeStrategy(FedAvg):
    def __init__(self, edge_groups, proximal_mu=0.0, edge_rounds=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #Starting edge groups from server_app.py
        #These are only used until KMeans has made the real groups
        self.edge_groups = edge_groups

        #For FedProx
        self.proximal_mu = proximal_mu

        #How many edge rounds before global aggregation
        self.edge_rounds = max(1, int(edge_rounds))

        # Number of clusters we want KMeans to make
        self.num_edges = len(edge_groups)

        #Stores latest model for each edge group
        self.edge_arrays = {}

        #Stores the latest global model
        self.global_arrays = None

        #Learns the Flower node_id -> partition_id after first reply
        self.node_to_partition = {}

        #KMeans should only run once (Updated now)
        self.clusters_frozen = False

        #Number of rounds we do k means clustering before mutiple agregations
        self.cluster_round = 10

        #Stores latest feature vector for each client before KMeans
        self.client_features = {}

        #All clients we expect to see before doing KMeans
        self.expected_clients = set()
        for group in edge_groups.values():
            for client_id in group:
                self.expected_clients.add(int(client_id))

        #Remember which round edge training actually starts
        self.first_edge_round = None


    def _edge_for_partition(self, partition_id: int):
        #Find which edge group a partition belongs to
        for edge_id, group in self.edge_groups.items():
            if partition_id in group:
                return edge_id
        return None


    def _weighted_average_arrayrecords(self, weighted_arrays):
        #Weighted average of edge models
        #weighted_arrays =  [(ArrayRecord, num_examples), ...]

        total_examples = sum(count for _, count in weighted_arrays)

        #Convert each ArrayRecord only once
        parsed_states = [
            (arrays.to_torch_state_dict(), count)
            for arrays, count in weighted_arrays
        ]

        first_state = parsed_states[0][0]
        avg_state = {}

        for key, first_value in first_state.items():

            #Only average float tensors
            if torch.is_floating_point(first_value):
                avg_tensor = torch.zeros_like(first_value, dtype=torch.float32)

                for state, count in parsed_states:
                    weight = count / total_examples
                    avg_tensor += weight * state[key].to(torch.float32)

                avg_state[key] = avg_tensor.to(dtype=first_value.dtype)

            #For things like BatchNorm num_batches_tracked
            else:
                avg_state[key] = first_value.clone()

        return ArrayRecord(avg_state)


    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Send edge model if we have one, otherwise send global model."""

        if self.fraction_train == 0.0:
            return []

        #Save first global model
        if self.global_arrays is None:
            self.global_arrays = arrays

        #FLOWER 1.29 FIX: sample nodes manually because we send different models
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        #Send FedProx mu to clients
        config["proximal-mu"] = self.proximal_mu
        config["server-round"] = server_round

        messages = []

        for node_id in node_ids:

            #We only know partition_id after client has replied once
            node_id_key = str(node_id)
            partition_id = self.node_to_partition.get(node_id_key)

            edge_id = (
                self._edge_for_partition(partition_id)
                if partition_id is not None
                else None
            )

            #If we know the client edge, send edge model
            if edge_id is not None and edge_id in self.edge_arrays:
                arrays_to_send = self.edge_arrays[edge_id]

                print(
                    f"[SERVER] Round {server_round}: sending EDGE model {edge_id} "
                    f"to partition {partition_id}",
                    flush=True,
                )

            #First round / unknown client gets global model
            else:
                arrays_to_send = self.global_arrays

                print(
                    f"[SERVER] Round {server_round}: sending GLOBAL model to node {node_id}",
                    flush=True,
                )

            record = RecordDict(
                {
                    "arrays": arrays_to_send,
                    "config": config,
                }
            )

            messages.append(
                Message(
                    content=record,
                    dst_node_id=node_id,
                    message_type=MessageType.TRAIN,
                )
            )

        return messages


    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:

        replies_list = list(replies)
        if not replies_list:
            return None, None

        #Client metrics
        train_losses = []
        train_accs = []

        #Used for KMeans first time
        client_ids = []
        feature_vectors = []
        valid_replies = []

        #STEP 1: collect replies
        for reply in replies_list:
            if not reply.has_content():
                print(f"Empty message received in round {server_round}. Skipping client.")
                continue

            metric_record = reply.content["metrics"]

            partition_id = int(metric_record["partition_id"])
            node_id = str(reply.metadata.src_node_id)

            #Save Flower node -> partition mapping
            self.node_to_partition[node_id] = partition_id

            client_ids.append(partition_id)
            feature_vectors.append(metric_record["feature_vector"])
            valid_replies.append(reply)

            #Save latest feature vector for this client
            self.client_features[partition_id] = metric_record["feature_vector"]

            num_examples = (
                int(metric_record["num-examples"])
                if "num-examples" in metric_record
                else 1
            )

            if "train_loss" in metric_record:
                train_losses.append((float(metric_record["train_loss"]), num_examples))

            if "train_acc" in metric_record:
                train_accs.append((float(metric_record["train_acc"]), num_examples))

            print(
                f"[SERVER] Round {server_round}: got update from client {partition_id}",
                flush=True,
            )

        if not valid_replies:
            return None, None

        #Before KMeans, just do normal global aggregation
        if not self.clusters_frozen and (
            server_round < self.cluster_round
            or not self.expected_clients.issubset(set(self.client_features.keys()))
        ):
            print(
                f"[SERVER] Round {server_round}: warmup round before KMeans. Doing normal global aggregation.",
                flush=True,
            )

            global_arrays, _ = super().aggregate_train(server_round, valid_replies)

            if global_arrays is None:
                return None, None

            self.global_arrays = global_arrays

            total_examples = sum(
                int(reply.content["metrics"]["num-examples"])
                for reply in valid_replies
            )

            aggregated_metrics: dict[str, float] = {
                "num-examples": total_examples,
            }

            if train_losses:
                aggregated_metrics["train_loss"] = (
                    sum(loss * n for loss, n in train_losses)
                    / sum(n for _, n in train_losses)
                )

            if train_accs:
                aggregated_metrics["train_acc"] = (
                    sum(acc * n for acc, n in train_accs)
                    / sum(n for _, n in train_accs)
                )

            return global_arrays, MetricRecord(aggregated_metrics)

        #STEP 2: run KMeans ONLY ONCE
        if not self.clusters_frozen and server_round >= self.cluster_round:
            historical_client_ids = sorted(self.client_features.keys())
            X = np.array([self.client_features[cid] for cid in historical_client_ids])

            num_clusters = min(self.num_edges, len(X))

            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=42,
                n_init="auto",
            )

            cluster_labels = kmeans.fit_predict(X)

            new_edge_groups = {edge_id: [] for edge_id in range(num_clusters)}

            for client_id, cluster_id in zip(historical_client_ids, cluster_labels):
                new_edge_groups[int(cluster_id)].append(int(client_id))

            self.edge_groups = new_edge_groups
            self.clusters_frozen = True
            self.first_edge_round = server_round

            print(
                f"\n[SERVER] KMeans done. Frozen edge groups: {self.edge_groups}",
                flush=True,
            )

            #Save cluster plot like before
            if len(X) > 1:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                plt.figure(figsize=(6, 4))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis")

                for i, cid in enumerate(historical_client_ids):
                    plt.annotate(f"C{cid}", (X_pca[i, 0], X_pca[i, 1]))

                plt.title(f"Round {server_round} Frozen KMeans Clusters")
                plt.savefig(f"experiment_fedtree/cluster_images/round_{server_round}_clusters.png")
                plt.close()

        else:
            print(
                f"[SERVER] Round {server_round}: using frozen edge groups {self.edge_groups}",
                flush=True,
            )

        #STEP 3: group replies by edge
        edge_replies = {edge_id: [] for edge_id in self.edge_groups.keys()}

        for reply in valid_replies:
            metric_record = reply.content["metrics"]
            partition_id = int(metric_record["partition_id"])

            edge_id = self._edge_for_partition(partition_id)

            if edge_id is None:
                print(
                    f"[SERVER] Round {server_round}: client {partition_id} has no edge group. Skipping.",
                    flush=True,
                )
                continue

            edge_replies[edge_id].append(reply)

        #STEP 4: aggregate inside each edge group
        edge_aggregates = []

        for edge_id, group_messages in edge_replies.items():
            if not group_messages:
                continue

            group_examples = sum(
                int(msg.content["metrics"]["num-examples"])
                for msg in group_messages
            )

            #FedAvg inside edge group
            edge_arrays, _ = super().aggregate_train(server_round, group_messages)

            if edge_arrays is not None:
                #Save edge model for next edge round
                self.edge_arrays[edge_id] = edge_arrays
                edge_aggregates.append((edge_arrays, group_examples))

                print(
                    f"[SERVER] Round {server_round}: updated EDGE model {edge_id} "
                    f"with {len(group_messages)} clients",
                    flush=True,
                )

        if not edge_aggregates:
            return None, None

        #STEP 5: metrics
        total_examples = sum(count for _, count in edge_aggregates)

        aggregated_metrics: dict[str, float] = {
            "num-examples": total_examples,
        }

        if train_losses:
            aggregated_metrics["train_loss"] = (
                sum(loss * n for loss, n in train_losses)
                / sum(n for _, n in train_losses)
            )

        if train_accs:
            aggregated_metrics["train_acc"] = (
                sum(acc * n for acc, n in train_accs)
                / sum(n for _, n in train_accs)
            )

        #STEP 6: global aggregation only every edge_rounds
        edge_round_number = server_round - self.first_edge_round + 1

        if edge_round_number % self.edge_rounds == 0:
            global_arrays = self._weighted_average_arrayrecords(edge_aggregates)

            self.global_arrays = global_arrays

            print(
                f"[SERVER] Round {server_round}: GLOBAL aggregation after "
                f"{self.edge_rounds} edge rounds",
                flush=True,
            )

            #After global aggregation, reset all edge models to global model
            for edge_id in self.edge_groups.keys():
                self.edge_arrays[edge_id] = global_arrays

            return global_arrays, MetricRecord(aggregated_metrics)

        #No global aggregation this round
        print(
            f"[SERVER] Round {server_round}: EDGE ONLY round. Keeping edge models separate.",
            flush=True,
        )

        return self.global_arrays, MetricRecord(aggregated_metrics)
    
class Scaffold(FedAvg):
    def __init__(self, initial_parameters: ArrayRecord, lr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.lr = lr

        """initialize control variate"""
        self.global_cv: dict[str, torch.Tensor] | None = None
  

    """configure next round - send global model and control variate to clients"""
    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:

        #Client sampling - same as FedAvg
        if self.fraction_train == 0.0:
            return []

        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        config["server-round"] = server_round

        self.global_state = arrays.to_torch_state_dict()
        #Save initial server param & set control variate to zero on first round
        if self.global_cv is None:
            state = arrays.to_torch_state_dict()
            self.global_cv = {key: torch.zeros_like(value) for key, value in state.items()}
        
        #add global cv to array record to send to clients 
        combined: dict[str, torch.Tensor] = dict(arrays.to_torch_state_dict())
        for key, value in self.global_cv.items():
            combined[f"__gcv__{key}"] = value

        #Construct message content with global model and control variate
        record = RecordDict(
            {
                "arrays": ArrayRecord(combined),
                "config": config,
            }
        )

        #Construct and return messages to clients
        return [Message(content=record, dst_node_id=node_id, message_type=MessageType.TRAIN,) for node_id in node_ids]
    
    """aggregate client updates - update global model and control variate"""
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]: #Flower 1.29 Fix: Use Tuple return type

        valid_replies = [reply for reply in replies if reply.has_content()]
        if not valid_replies:
            return None, None


        #separate model updates & cv diff from client messages
        cv_difference: list[dict[str, torch.Tensor]] = []
        model_states:list[dict[str, torch.Tensor]] = []
        num_examples_list: list[int] = []
        train_losses: list[float] = []
        train_accs: list[float] = []


        for reply in valid_replies:
            combined = reply.content["arrays"].to_torch_state_dict()
            cv_diff = {key[len("__cv__"):]: value 
                       for key, value in combined.items() if key.startswith("__cv__")}
            model_state = {key: value 
                           for key, value in combined.items() if not key.startswith("__cv__")}
            cv_difference.append(cv_diff)
            model_states.append(model_state)
            num_examples_list.append(
                int(reply.content["metrics"]["num-examples"])
                if "num-examples" in reply.content["metrics"] else 1)
            
            # Extract train_loss and train_acc from client metrics
            if "train_loss" in reply.content["metrics"]:
                train_losses.append((float(reply.content["metrics"]["train_loss"]), int(reply.content["metrics"]["num-examples"])))
            if "train_acc" in reply.content["metrics"]:
                train_accs.append((float(reply.content["metrics"]["train_acc"]), int(reply.content["metrics"]["num-examples"])))

        #aggregate client control variates into global control variate update
        if cv_difference and self.global_cv is not None:
            total_clients = len(cv_difference)
            sampled_clients = len(valid_replies)
            with torch.no_grad():
                for key in self.global_cv.keys():                                                          #loop through each layer of the model
                    total_cv_diff = (1/sampled_clients)*torch.stack([cv_diff[key] for cv_diff in cv_difference]).sum(dim=0)    #sum up the control variate differences for this layer across clients
                    self.global_cv[key] = self.global_cv[key] + (sampled_clients / total_clients)*total_cv_diff            #update global control variate by adding average client control variate difference

        #aggregate client model updates (cant do fedavg anymore bc control variate is in the same array record)
        total_examples = sum(num_examples_list)
        if total_examples == 0:
            return None, None
        
        #avg_state = {key: torch.zeros_like(value, dtype=torch.float32)
        #             for key, value in model_states[0].items()}
        new_global = {key: torch.zeros_like(value, dtype=torch.float32)
                     for key, value in model_states[0].items()}
        #Calculate the new model
        with torch.no_grad():
            for key in self.global_state.keys():
                model_diff = (1/sampled_clients)*(torch.stack([model[key] - self.global_state[key] for model in model_states]).sum(dim=0))
                new_global[key] = self.global_state[key] + self.lr*model_diff
        
        '''
        for state, num_examples in zip(model_states, num_examples_list):
            weight = num_examples / total_examples
            for key in avg_state:
                avg_state[key] += weight * state[key].to(torch.float32)
        '''
        #Weighted average aggregation for train_loss and train_acc
        aggregated_metrics: dict[str, float] = {"num-examples": total_examples}

        if train_losses:
            aggregated_metrics["train_loss"] = (
                sum(loss * n for loss, n in train_losses) / sum(n for _, n in train_losses)
            )
        if train_accs:
            aggregated_metrics["train_acc"] = (
                sum(acc * n for acc, n in train_accs) / sum(n for _, n in train_accs)
            )

        return ArrayRecord(new_global), MetricRecord(aggregated_metrics)


class FedAvgCyclic(FedAvg):
    def __init__(self, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, weighted_by_key = "num-examples", arrayrecord_key = "arrays", configrecord_key = "config", train_metrics_aggr_fn = None, evaluate_metrics_aggr_fn = None, seed = 0):
        super().__init__(fraction_train, fraction_evaluate, min_train_nodes, min_evaluate_nodes, min_available_nodes, weighted_by_key, arrayrecord_key, configrecord_key, train_metrics_aggr_fn, evaluate_metrics_aggr_fn)
        np.random.seed(seed)

        self.thread_to_local_models = {}
        self.thread_targets = {}
        self.thread_to_client = {}

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = client_manager.all()
        sorted_cids = sorted(all_clients.keys())
        print(sorted_cids)
        #self.num_clients = len(all_clients)
        if(self.num_clients == None):
            self.num_clients = len(sorted_cids)
        num_of_threads = max(math.floor(self.fraction_fit*self.num_clients),1)
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        if(server_round % self.num_clients == 0):
            selected_clients = np.random.choice(sorted_cids,num_of_threads, replace=False)
            for client in selected_clients:
                self.thread_to_local_models[client] = parameters
                self.thread_to_client[client] = client
        
        ins = []
        

        for thread_id, target_cid in self.thread_to_client.items():
            target_cid = str(int(target_cid) % self.num_clients)
            cid = sorted_cids[target_cid]
            client_proxy = all_clients[cid]
            print(f"thread {thread_id} -> client {target_cid}")

            fit_ins = FitIns(self.thread_to_local_models[thread_id],config)
            ins.append((client_proxy, fit_ins))

        return ins
    
    def aggregate_fit(self, server_round, results, failures):
        
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            tid = value_to_key(cid, self.thread_to_client)
            if tid != None:
                self.thread_to_local_models[tid] = fit_res.parameters
                self.thread_to_client[tid] = str(int(self.thread_to_client) + 1) 
        
        if server_round % self.num_clients != 0 :
            return None, {}

        weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), 1)
                for _, fit_res in results
            ]
        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
    
def value_to_key(str, dict):
    ret = None
    for key in dict :
        if(dict[key] == str):
            ret = key
    return ret