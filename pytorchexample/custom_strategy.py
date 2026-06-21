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
    def __init__(self, edge_groups, proximal_mu=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_groups = edge_groups
        self.proximal_mu = proximal_mu

    #A bunch of stuff had to change to fit flower 1.29
    def configure_train(
        self, 
        server_round: int, 
        arrays: ArrayRecord, 
        config: ConfigRecord, 
        grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        #print(f"------------- Round {server_round}: Configuring training -------------", flush=True)

        config["proximal-mu"] = self.proximal_mu

        print(f"[SERVER] Round {server_round}: sending proximal_mu={self.proximal_mu}",flush=True,)

        return super().configure_train(
            server_round=server_round,
            arrays=arrays,
            config=config,
            grid=grid
        )

    def aggregate_train(
        self, 
        server_round: int, 
        replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        
        replies_list = list(replies)
        if not replies_list:
            return None, None
        
        #Lists to hold the client metrics (train loss and acc)
        train_losses: list[float] = []
        train_accs: list[float] = []

        # STEP 1: EXTRACT FEATURE VECTORS
        client_ids = []
        feature_vectors = []
        valid_replies = [] 

        for reply in replies_list:
            if not reply.has_content():
                print(f"Empty message received in round {server_round}. Skipping client.")
                continue

            #Access the metrics namespace strictly
            metric_record = reply.content["metrics"]
            
            client_ids.append(int(metric_record["partition_id"]))
            feature_vectors.append(metric_record["feature_vector"])
            valid_replies.append(reply)

            #Extracting metrics for global weighted average
            num_examples = int(metric_record["num-examples"]) if "num-examples" in metric_record else 1
            if "train_loss" in metric_record:
                train_losses.append((float(metric_record["train_loss"]), num_examples))
            if "train_acc" in metric_record:
                train_accs.append((float(metric_record["train_acc"]), num_examples))

            #Print the round, feature vector len and the client its coming from
            pid = metric_record["partition_id"]
            fv_len = len(metric_record["feature_vector"])
            print(f"[DEBUG - Server] Round {server_round}: Received feature vector of length {fv_len} from Client {pid}", flush=True)

        if not valid_replies:
            return None, None

        #STEP 2: RUN K-MEANS CLUSTERING
        X = np.array(feature_vectors)
        num_clusters = min(2, len(X))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X)

        #STEP 3: CREATE NEW EDGE GROUPS
        new_edge_groups = {0: [], 1: []}
        for client_id, cluster_id in zip(client_ids, cluster_labels):
            new_edge_groups[int(cluster_id)].append(int(client_id))
        
        self.edge_groups = new_edge_groups

        if len(X) > 1:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            plt.figure(figsize=(6, 4))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
            for i, cid in enumerate(client_ids):
                plt.annotate(f"C{cid}", (X_pca[i, 0], X_pca[i, 1]))
            plt.title(f"Round {server_round} Clusters")
            plt.savefig(f"experiment_fedtree/cluster_images/round_{server_round}_clusters.png")
            plt.close()

        print(f"\nNew clustered edge groups: {self.edge_groups}")

        # STEP 4: GROUP REPLIES BY EDGE-SERVERS
        edge_replies = {0: [], 1: []}
        for reply in valid_replies:
           
            metric_record = reply.content["metrics"]
            partid = int(metric_record["partition_id"]) 
            
            for edge_id, group in self.edge_groups.items():
                if partid in group:
                    edge_replies[edge_id].append(reply)
                    break

        #STEP 5 + 6: AGGREGATE PER EDGE AND THEN GLOBALLY
        edge_aggregates = []
        for edge_id, group_messages in edge_replies.items():
            if not group_messages:
                continue
            
            #FLOWER 1.29 FIX: Access namespace
            group_examples = sum(int(msg.content["metrics"]["num-examples"]) for msg in group_messages)
            
            edge_arrays, _ = super().aggregate_train(server_round, group_messages)
            
            if edge_arrays is not None:
                edge_aggregates.append((edge_arrays, group_examples))

        if not edge_aggregates:
            return None, None

        #STEP 6: GLOBAL WEIGHTED AVERAGE
        total_examples = sum(count for _, count in edge_aggregates)
        
        first_arrays, _ = edge_aggregates[0]
        avg_state = {k: torch.zeros_like(v, dtype=torch.float32) 
                     for k, v in first_arrays.to_torch_state_dict().items()}

        for arrays, count in edge_aggregates:
            weight = count / total_examples
            client_state = arrays.to_torch_state_dict()
            for key in avg_state:
                avg_state[key] += weight * client_state[key].to(torch.float32)

        aggregated_metrics: dict[str, float] = {"num-examples": total_examples}

        if train_losses:
            aggregated_metrics["train_loss"] = (
                sum(loss * n for loss, n in train_losses) / sum(n for _, n in train_losses)
            )
        if train_accs:
            aggregated_metrics["train_acc"] = (
                sum(acc * n for acc, n in train_accs) / sum(n for _, n in train_accs)
            )

        return ArrayRecord(avg_state), MetricRecord(aggregated_metrics)
    
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