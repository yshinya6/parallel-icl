import pdb

import torch
import torch.nn.functional as F


def reduce_images(image_features, query_feature, reducer, num_images):
    match reducer:
        case "default":
            return default_drop(image_features, num_images)
        case "random":
            return random_drop(image_features, num_images)
        case "div_prune":
            return div_prune(image_features, num_images)
        case "max_similarity":
            return max_similarity(image_features, query_feature, num_images)
        case "small_norm":
            return small_norm(image_features, num_images)
        case "large_norm":
            return large_norm(image_features, num_images)


def default_drop(context_features, num_images):
    valid_indices = torch.arange(0, num_images, device=context_features.device, dtype=torch.long)
    reduced_embeddings = context_features[valid_indices]
    return reduced_embeddings, valid_indices


def random_drop(context_features, num_images):
    batch_dim = context_features.shape[0]
    shuffled_indices = torch.randperm(batch_dim, device=context_features.device)
    valid_indices = shuffled_indices[:num_images]
    valid_indices, _ = torch.sort(valid_indices)
    reduced_embeddings = context_features[valid_indices]
    return reduced_embeddings, valid_indices


def max_similarity(context_features, query_feature, num_images):
    if query_feature.dim() == 1:
        query_feature = query_feature.unsqueeze(0)

    normalized_query = F.normalize(query_feature, p=2, dim=1)
    normalized_context = F.normalize(context_features, p=2, dim=1)

    cos_sim = torch.matmul(normalized_query, normalized_context.transpose(0, 1)).squeeze()
    _, top_k_indices = torch.topk(cos_sim, k=num_images, largest=True)
    top_k_features = context_features[top_k_indices]
    return top_k_features, top_k_indices


def _pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity


def div_prune(context_features, num_images):
    cosine_matrix = 1.0 - (_pairwise_cosine_similarity(context_features))

    selected_index = torch.empty(num_images, dtype=torch.long, device=context_features.device)
    for i in range(num_images):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(
                cosine_matrix, 0, torch.index_select(selected_index, 0, torch.arange(0, i, device=cosine_matrix.device))
            )

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]  # for distance
        else:
            scores = torch.min(m2, dim=0).values  # for distance

        phrase_to_add_idx = torch.argmax(scores)
        selected_index[i] = phrase_to_add_idx
    return context_features[selected_index], selected_index


def small_norm(context_features: torch.Tensor, num_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.norm(context_features, p=2, dim=1)
    sorted_indices = torch.argsort(norms)
    valid_indices = sorted_indices[:num_images]
    valid_indices, _ = torch.sort(valid_indices)
    reduced_tensor = context_features[valid_indices, :]
    return reduced_tensor, valid_indices


def large_norm(context_features: torch.Tensor, num_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.norm(context_features, p=2, dim=1)
    sorted_indices = torch.argsort(norms, descending=True)
    valid_indices = sorted_indices[:num_images]
    valid_indices, _ = torch.sort(valid_indices)
    reduced_tensor = context_features[valid_indices, :]
    return reduced_tensor, valid_indices
