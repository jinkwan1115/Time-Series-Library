import torch

def compute_cka_torch(repr_pred, repr_target, device):
    """
    Parameters:
    - repr_pred: feature matrix of Y_hat (repr_length x num_variables)
    - repr_target: feature matrix of Y (repr_length x num_variables)
    
    Returns:
    - CKA similarity value (scalar)
    """
    # Compute the linear kernel
    K_pred = torch.mm(repr_pred, repr_pred.T).to(device)
    K_target = torch.mm(repr_target, repr_target.T).to(device)
    
    # Center the kernels
    def center_kernel(kernel):
        n = kernel.size(0)
        one_n = torch.ones((n, n), device=kernel.device) / n
        return kernel - one_n @ kernel - kernel @ one_n + one_n @ kernel @ one_n

    K_pred_centered = center_kernel(K_pred)
    K_target_centered = center_kernel(K_target)
    
    # Compute the numerator and denominator for CKA
    numerator = torch.sum(K_pred_centered * K_target_centered)
    denominator = torch.sqrt(torch.sum(K_pred_centered ** 2) * torch.sum(K_target_centered ** 2))
    
    # Avoid division by zero
    if denominator == 0:
        return torch.tensor(0.0, device=repr_pred.device)
    
    return numerator / denominator

def cka_torch(repr_pred_batch, repr_target_batch, device):
    """
    Parameters:
    - repr_pred_batch: Batch feature matrix 1 (batch_size x repr_length x num_variables, on GPU)
    - repr_target_batch: Batch feature matrix 2 (batch_size x repr_length x num_variables, on GPU)
    
    Returns:
    - CKA loss (scalar)
    """
    batch_size = repr_pred_batch.size(0)
    cka_values = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        repr_pred = repr_pred_batch[b].to(device)  # Shape: (repr_length, num_variables)
        repr_target = repr_target_batch[b].to(device)  # Shape: (repr_length, num_variables)
        cka_value = compute_cka_torch(repr_pred, repr_target, device)
        cka_values[b] = cka_value
    
    cka_loss = 1.0 - torch.mean(cka_values)
    
    return cka_loss