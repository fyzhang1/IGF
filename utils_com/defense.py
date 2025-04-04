import torch
import torch.nn.functional as F

def perturb_gradient(gradient, sensitivity_factor=0.1, perturbation_scale=0.01):
    """
    Perturbs gradient by adding more noise to dimensions with higher sensitivity.
    
    Args:
        gradient: The original gradient tensor
        sensitivity_factor: Factor determining how much to scale perturbation by gradient magnitude
        perturbation_scale: Base scale of perturbation
    
    Returns:
        Perturbed gradient tensor
    """
    # Calculate sensitivity of each dimension based on gradient magnitude
    sensitivity = gradient.abs() * sensitivity_factor
    
    # Generate perturbation noise
    perturbation = torch.randn_like(gradient) * perturbation_scale
    
    # Scale perturbation by sensitivity
    scaled_perturbation = perturbation * sensitivity
    
    # Add perturbation to gradient
    perturbed_gradient = gradient + scaled_perturbation
    
    return perturbed_gradient



def smooth_gradient(gradient, smoothing_factor=0.1):
    """
    Smooths the gradient using a simpler approach that works with any tensor shape.
    
    Args:
        gradient: The original gradient tensor
        smoothing_factor: Factor determining smoothing strength (0 to 1)
    
    Returns:
        Smoothed gradient tensor
    """
    # Store original shape for later reshape
    original_shape = gradient.shape
    
    # Reshape to 2D: (batch_size, features)
    batch_size = original_shape[0]
    flattened = gradient.view(batch_size, -1)
    
    # Apply a simple smoothing approach - moving average along features
    smoothed = torch.zeros_like(flattened)
    
    # Use a simple rolling average for each sample
    window_size = 5
    padded = F.pad(flattened, (window_size//2, window_size//2))
    
    for i in range(flattened.size(1)):
        # Get window for this position
        window = padded[:, i:i+window_size]
        # Average values in the window
        smoothed[:, i] = torch.mean(window, dim=1)
    
    # Blend original and smoothed gradients
    blended = (1 - smoothing_factor) * flattened + smoothing_factor * smoothed
    
    # Reshape back to original shape
    return blended.view(original_shape)