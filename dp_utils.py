import torch
import geoopt 
import math 
import numpy as np


def compute_gradient_importance(grad):

    l1_norm = torch.abs(grad).mean(dim=-1)

    l2_norm = torch.norm(grad, p=2, dim=-1)

    sparsity = (grad.abs() > 1e-5).float().mean(dim=-1)

    importance = (l1_norm + l2_norm) * sparsity

    if importance.max() == importance.min():
        return torch.ones_like(importance)
    normalized_importance = (importance - importance.min()) / (importance.max() - importance.min())
    
    return normalized_importance

def clip_gradients(grads, max_norm):
    """Clips the L2 norm of a list of gradient tensors."""
    total_norm = 0.0
    non_none_grads = [grad for grad in grads if grad is not None]
    if not non_none_grads:
        return grads # Return original list if all are None

    for grad in non_none_grads:
        param_norm = grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clipped_grads = []
    if clip_coef < 1:
        for grad in grads: # Iterate original list to preserve None structure
             if grad is None: 
                 clipped_grads.append(None)
                 continue
             clipped_grads.append(grad.data.mul(clip_coef))
    else:
        # If norm is already below max_norm, just clone original data
        for grad in grads:
            if grad is None:
                clipped_grads.append(None)
                continue
            clipped_grads.append(grad.data.clone()) 
            
    return clipped_grads # Return new list of tensors


# DRO Utilities for sample reweighting
def compute_sample_weights(loss_values, dro_params):
    """
    Compute sample weights using Chi-squared DRO.
    
    Args:
        loss_values: Per-sample loss values
        dro_params: DRO hyperparameters (chi_square_rho, etc.)
        
    Returns:
        Weights for each sample
    """
    eps = 1e-12
    rho = dro_params['chi_square_rho']
    # Normalize loss values
    normalized_losses = loss_values / (loss_values.mean() + eps)
    
    # Chi-square DRO weights computation
    weights = (1 + rho * normalized_losses).detach()
    weights = weights / (weights.sum() + eps)
    
    return weights


def sigmoid_scale(x, center_radius, steepness):
    """Sigmoid function for continuous noise scaling.
    
    Args:
        x: Input value (hyperbolic radius)
        center_radius: Center point of sigmoid (where scaling is 0.5)
        steepness: Controls how quickly the scaling changes (higher = sharper transition)
    
    Returns:
        Scaling factor between 0 and 1
    """
    return 1 / (1 + torch.exp(-steepness * (x - center_radius)))

def compute_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:

    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("epsilon > 0, 0 < delta < 1")
    if sensitivity <= 0:
        sensitivity = 1
    

    sigma = (math.sqrt(2 * math.log(1.25 / delta)) * sensitivity) / epsilon
    return sigma

def compute_l2_sensitivity(original_grad, perturbed_grad):
    return np.linalg.norm(original_grad.detach().numpy() - perturbed_grad.detach().numpy(), ord=2)

def get_noise(gw_real, gw_syn, args, manifold, device):
    epsilon = args.current_epsilon
    delta = args.target_delta

    sensitivity = 1.0
    base_sigma = compute_gaussian_sigma(epsilon, delta, sensitivity)

    grad_importance = compute_gradient_importance(gw_real)
    importance_factor = 1.0 - grad_importance 

    # Calculate hyperbolic radii and sigmoid scaling
    radii = manifold.dist0(gw_real)
    scaling_factors = sigmoid_scale(radii, center_radius=0.5, steepness=1.0)

    grad_norm = torch.norm(gw_real, p=2, dim=-1, keepdim=True).mean()

    if grad_norm < 1e-6:
        grad_norm = 1e-6  

    sigma = base_sigma *  (grad_norm / (sensitivity + 1e-8))  
    sigma = max(sigma, 1e-6)  
    sigma = min(sigma, 10.0 * sensitivity / epsilon)  

    noise = torch.randn_like(gw_real, device=device) * sigma * importance_factor.reshape(-1, 1) * (0.8 + 0.4 * scaling_factors).reshape(-1, 1)
    return noise
