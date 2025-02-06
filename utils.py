import os
import random

import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

# Display original and reconstructed frames side-by-side
def visualize_frames(original, reconstructed):
    for i in range(min(original.size(0), 8)):  # Show up to 8 frames
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(TF.to_pil_image(original[i].cpu()))
        axs[0].set_title("Original Frame")
        axs[0].axis('off')
        axs[1].imshow(TF.to_pil_image(reconstructed[i].squeeze(0).cpu().clamp(0, 1)))
        axs[1].set_title("Reconstructed Frame")
        axs[1].axis('off')
        plt.show()


# Calculate metrics for pixel space
def calculate_metrics_pixel_space_frame_2_frame(frames):
    pixel_psnr = []
    pixel_ssim = []
    for i in range(len(frames) - 1):
        frame1, frame2 = frames[i].detach().numpy(), frames[i + 1].detach().numpy()
        # Compute PSNR
        psnr_value = psnr(frame1, frame2, data_range=frame1.max() - frame1.min())
        pixel_psnr.append(psnr_value)
        # Compute SSIM with appropriate win_size
        frame1_ssim = np.transpose(np.squeeze(frame1), (1, 2, 0))
        frame2_ssim = np.transpose(np.squeeze(frame2), (1, 2, 0))

        h, w, c = frame1_ssim.shape  # Get spatial dimensions and channels
        win_size = min(h, w)  # Use the smaller dimension for win_size

        # Ensure win_size is odd and valid
        if win_size % 2 == 0:
            win_size -= 1

        ssim_value = ssim(frame1_ssim, frame2_ssim, data_range=frame1_ssim.max() - frame2_ssim.min(), win_size=win_size, channel_axis=-1)
        pixel_ssim.append(ssim_value)

    print(f"Average PSNR: {sum(pixel_psnr) / len(pixel_psnr)}")
    print(f"Average SSIM: {sum(pixel_ssim) / len(pixel_ssim)}")
    return sum(pixel_psnr) / len(pixel_psnr), sum(pixel_ssim) / len(pixel_ssim)


def calculate_metrics_latent_space(latent_vectors):
    euclidean_distances = []
    cosine_similarities = []
    for i in range(len(latent_vectors) - 1):
        z1, z2 = latent_vectors[i], latent_vectors[i + 1]

        # Flatten the tensors
        z1_flattened = torch.flatten(z1).unsqueeze(0).double() # give them a batch dimension
        z2_flattened = torch.flatten(z2).unsqueeze(0).double() # turn them to double for better precision

        euclidean_distances.append(F.pairwise_distance(z1_flattened.unsqueeze(0), z2_flattened.unsqueeze(0)).item())
        cosine_similarities.append(F.cosine_similarity(z1_flattened, z2_flattened, dim=1).item())

    print(f"Average Euclidean Distance: {sum(euclidean_distances) / len(euclidean_distances)}")
    print(f"Average Cosine Similarity : {sum(cosine_similarities) / len(cosine_similarities)}")
    return sum(euclidean_distances) / len(euclidean_distances), sum(cosine_similarities) / len(cosine_similarities)

def z_score_normalization(latent_vectors):
    """
    Apply Z-score normalization to a batch of latent vectors.
    Args:
        latent_vectors (numpy.ndarray or torch.Tensor): shape (N, D)
    Returns:
        Normalized latent vectors with mean 0 and std 1 per dimension.
    """
    latent_vectors = np.array(latent_vectors)
    mean = latent_vectors.mean(axis=0, keepdims=True)
    std = latent_vectors.std(axis=0, keepdims=True) + 1e-8  # Avoid division by zero
    return torch.from_numpy((latent_vectors - mean) / std)

def unit_norm_normalization(latent_vectors):
    """
    Apply unit norm normalization to a batch of latent vectors.
    Args:
        latent_vectors (numpy.ndarray or torch.Tensor): shape (N, D)
    Returns:
        Normalized latent vectors with unit norm.
    """
    latent_vectors = np.array(latent_vectors)
    norm = np.linalg.norm(latent_vectors, axis=1, keepdims=True) if isinstance(latent_vectors, np.ndarray) else torch.norm(latent_vectors, dim=1, keepdim=True)
    return torch.from_numpy(latent_vectors / (norm + 1e-8))  # Avoid division by zero