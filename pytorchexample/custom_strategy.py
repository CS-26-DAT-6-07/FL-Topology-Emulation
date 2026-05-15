from sklearn.cluster import KMeans
import numpy as np

from flwr.serverapp.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


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


