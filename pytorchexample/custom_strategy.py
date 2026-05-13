import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Iterable, List, Tuple, Optional

from flwr.app import Message, ArrayRecord, MetricRecord, ConfigRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg 
from sklearn.decomposition import PCA

STRATEGY_NAME = "Tree_Custom_Strategy"

class CustomStrategy(FedAvg):
    def __init__(self, edge_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_groups = edge_groups

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
    
        # THIS LINE MUST BE INDENTED
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
        
        #print("------------CUSTOM aggregate_train CALLED--------------", flush=True)
        
        # Convert to list so we can iterate multiple times
        replies_list = list(replies)
        if not replies_list:
            return None, None

        # STEP 1: EXTRACT FEATURE VECTORS
        client_ids = []
        feature_vectors = []
        valid_replies = [] #Keep track of replies that didn't crash

        for reply in replies_list:

            #Check if the message has content (it won't if the client crashed)
            if not reply.has_content():
                print(f"Empty message received in round {server_round}. Skipping client.")
                continue

            metrics = reply.content.get("metrics", {})
            client_ids.append(int(metrics.get("partition_id", 0)))
            feature_vectors.append(metrics.get("feature_vector", []))
            valid_replies.append(reply)

        if not valid_replies:
            return None, None

        # STEP 2: RUN K-MEANS CLUSTERING
        X = np.array(feature_vectors)
        num_clusters = min(2, len(X))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X)

        # STEP 3: CREATE NEW EDGE GROUPS
        new_edge_groups = {0: [], 1: []}
        for client_id, cluster_id in zip(client_ids, cluster_labels):
            new_edge_groups[int(cluster_id)].append(int(client_id))
        
        self.edge_groups = new_edge_groups

        #Only plot if we have more than 1 client
        if len(X) > 1:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            plt.figure(figsize=(6, 4))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
            for i, cid in enumerate(client_ids):
                plt.annotate(f"C{cid}", (X_pca[i, 0], X_pca[i, 1]))
            plt.title(f"Round {server_round} Clusters")
            plt.savefig(f"round_{server_round}_clusters.png")
            plt.close()

        print(f"\nNew clustered edge groups: {self.edge_groups}") #Print the edge groups every round

        #STEP 4: GROUP REPLIES BY EDGE SERVER
        edge_replies = {0: [], 1: []}
        for reply in valid_replies:
            partid = int(reply.content["metrics"]["partition_id"])
            for edge_id, group in self.edge_groups.items():
                if partid in group:
                    edge_replies[edge_id].append(reply)
                    break

        #STEP 5 & 6: AGGREGATE PER EDGE AND THEN GLOBALLY
        edge_aggregates = []
        for edge_id, group_messages in edge_replies.items():
            if not group_messages:
                continue
            
            # 1. Calculate the total examples for THIS edge group manually
            group_examples = sum(int(msg.content["metrics"].get("num-examples", 0)) for msg in group_messages)
            
            edge_arrays, _ = super().aggregate_train(server_round, group_messages)
            
            if edge_arrays is not None:
                # 2. Store the arrays and the manual count
                edge_aggregates.append((edge_arrays, group_examples))

        if not edge_aggregates:
            return None, None

        # STEP 6 (Cont.): GLOBAL WEIGHTED AVERAGE
        total_examples = sum(count for _, count in edge_aggregates)
        
        first_arrays, _ = edge_aggregates[0]
        avg_state = {k: torch.zeros_like(v, dtype=torch.float32) 
                     for k, v in first_arrays.to_torch_state_dict().items()}

        for arrays, count in edge_aggregates:
            weight = count / total_examples
            client_state = arrays.to_torch_state_dict()
            for key in avg_state:
                avg_state[key] += weight * client_state[key].to(torch.float32)

        return ArrayRecord(avg_state), MetricRecord({"num-examples": total_examples})