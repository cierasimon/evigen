import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def query_similarity_penalty(queries, temperature):
    # queries = F.normalize(queries, p=2, dim=-1)
    distance_matrix = torch.cdist(queries, queries, p=2)
    similarity_matrix = torch.exp(-distance_matrix / temperature)
    return torch.norm(similarity_matrix, p="fro")


def query_distance_penalty(queries, threshold):
    out = F.normalize(queries, p=2, dim=1)
    distance_matrix = torch.cdist(out, out, p=2)
    return F.relu(distance_matrix - threshold).mean()


class QueryLoss(nn.Module):
    def __init__(self, temperature: float, ignore_index: int) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.temperature = temperature

    def forward(self, input: Tensor, label: Tensor, query: Tensor) -> Tensor:
        penalty = query_similarity_penalty(query, self.temperature)
        return self.cross_entropy(input, label) + penalty
