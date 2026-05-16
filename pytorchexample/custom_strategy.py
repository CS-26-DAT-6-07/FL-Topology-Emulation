from sklearn.cluster import KMeans
import numpy as np
import math

from collections.abc import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters,FitIns
from flwr.server.strategy.aggregate import aggregate


class TreeStrategy(FedAvg):
    #edge_groups is which clients belong to which edge server
    def __init__(self, edge_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_groups = edge_groups

    #This currently just uses normal FedAvg client selection (Can change later)
    #Runs BEFORE clients train
    def configure_fit(self, server_round, arrays, config, grid):
        return super().configure_fit(server_round, arrays, config, grid)

    #Runs AFTER all selected clients finish training
    def aggregate_fit(self, server_round, results, failures):

        #STEP 1: EXTRACT FEATURE VECTORS FROM CLIENT RESULTS
        client_ids = []
        feature_vectors = []

        for client, fit_res in results:

            #Metrics sent from client_app.py (Mesage)
            metrics = fit_res.metrics

            #Client partition id
            client_ids.append(metrics["partition_id"])

            #Hidden layer feature vector from client
            feature_vectors.append(metrics["feature_vector"])

        #Convert to numpy array for sklearn
        X = np.array(feature_vectors)

        #STEP 2: RUN K-MEANS CLUSTERING
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto",)
        cluster_labels = kmeans.fit_predict(X)

        #STEP 3: CREATE NEW EDGE GROUPS FROM CLUSTERS
        new_edge_groups = {0: [], 1: []}

        for client_id, cluster_id in zip(client_ids, cluster_labels):
            new_edge_groups[int(cluster_id)].append(int(client_id))

        #Update strategy edge groups dynamically
        self.edge_groups = new_edge_groups

        print("\nNew clustered edge groups:")
        print(self.edge_groups)

        #STEP 4: GROUP CLIENT RESULTS BY EDGE SERVER

        edge_results = {0: [], 1: []}

        for client, fit_res in results:

            partid = client.node_config["partition-id"]

            for edge_id, group in self.edge_groups.items():

                if partid in group:
                    edge_results[edge_id].append((client, fit_res))
                    break

        #STEP 5: RUN FEDAVG INSIDE EACH EDGE SERVER
        edge_aggregates = []

        for edge_id, group_results in edge_results.items():

            #Skip empety groups
            if len(group_results) == 0:
                continue

            # Run normal FedAvg on ONLY this edge group
            edge_arrays, edge_metrics = super().aggregate_fit(
                server_round,
                group_results,
                [],
            )

            #Store edge-level aggregated model
            edge_aggregates.append((edge_arrays, edge_metrics))

        # Safety check
        if not edge_aggregates:
            return None, {}


        #STEP 6: EXTRACT EDGE MODELS + THEIR WEIGHTS
        arrays_list = []
        weights = []

        for arrays, metrics in edge_aggregates:

            #Convert Flower parameters -> numpy arrays
            arrays_list.append(parameters_to_ndarrays(arrays))

            #Number of examples used in this edge
            weights.append(metrics["num-examples"])

        #STEP 7: GLOBAL AGGREGATION OF EDGE MODELS
        avg_ndarrays = []

        #Loop through each neural network layer
        for layer_idx in range(len(arrays_list[0])):

            #Weighted average of the layer across edge models
            layer_sum = sum(
                w * edge_model[layer_idx]
                for w, edge_model in zip(weights, arrays_list)
            )

            avg_ndarrays.append(layer_sum / sum(weights))

        #STEP 8: CONVERT BACK TO FLOWER FORMAT
        global_arrays = ndarrays_to_parameters(avg_ndarrays)

        #Return new global model
        return global_arrays, {
            "num-examples": sum(weights)
        }


class Scaffold(FedAvg):
    def __init__(self, initial_parameters: ArrayRecord, lr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.lr = lr

        """initialize control variate"""
        self.global_cv: Optional[dict[str, torch.Tensor]] = None
  

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

        #Save initial server param & set control variate to zero on first round
        if self.global_cv is None:
            state = arrays.to_torch_state_dict()
            self.global_cv = {key: torch.zeros_like(value) for key, value in state.items()}

        #Construct message content with global model and control variate
        record = RecordDict(
            {
                "arrays": self.initial_parameters,
                "config": config,
                "global_cv": ArrayRecord(self.global_cv),
            }
        )

        #Construct and return messages to clients
        return self._construct_messages(record, node_ids, MessageType.TRAIN)
    
    """aggregate client updates - update global model and control variate"""
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> ArrayRecord:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        #build client control variate update dict
        cv_difference: list[dict[str, torch.Tensor]] = [
            reply.content["control_variate"].to_torch_state_dict() 
            for reply in valid_replies
            if "control_variate" in reply.content
        ]

        #aggregate client control variates into global control variate update
        if cv_difference and self.global_cv is not None:
            total_clients = len(cv_difference)
            with torch.no_grad():
                for key in self.global_cv.keys():                                                          #loop through each layer of the model
                    total_cv_diff = torch.stack([cv_diff[key] for cv_diff in cv_difference]).sum(dim=0)    #sum up the control variate differences for this layer across clients
                    self.global_cv[key] = self.global_cv[key] + total_cv_diff / total_clients              #update global control variate by adding average client control variate difference

        #aggregate client model updates with FedAvg
        return super().aggregate_train(server_round, replies)


class FedAvgCyclic(FedAvg):
    def __init__(self, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, weighted_by_key = "num-examples", arrayrecord_key = "arrays", configrecord_key = "config", train_metrics_aggr_fn = None, evaluate_metrics_aggr_fn = None):
        super().__init__(fraction_train, fraction_evaluate, min_train_nodes, min_evaluate_nodes, min_available_nodes, weighted_by_key, arrayrecord_key, configrecord_key, train_metrics_aggr_fn, evaluate_metrics_aggr_fn)


        self.micro_round = 0
        self.thread_to_local_models = {}
        self.thread_targets = {}
        self.thread_to_client = {}

    


    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = client_manager.all()
        sorted_cids = sorted(all_clients.keys())
        self.num_clients = len(all_clients)
        if(self.num_clients == None):
            self.num_clients = len(sorted_cids)
        num_of_threads = math.floor(self.fraction_fit*self.num_clients)
        if num_of_threads == 0:
            raise ValueError("fraction_fit must result in a non zero number of clients")

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
            target_cid = target_cid % self.num_clients
            cid = sorted_cids[target_cid]
            client_proxy = all_clients[cid]

            fit_ins = FitIns(self.thread_to_local_models[thread_id],config)
            ins.append((client_proxy, fit_ins))

        return ins
    
    def aggregate_fit(self, server_round, results, failures):
        
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            tid = value_to_key(cid, self.thread_to_client)
            if tid != None:
                self.thread_to_local_models[tid] = fit_res.parameters
                self.thread_to_client[tid] += 1

        self.micro_round += 1
        
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