import pandas as pd
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from tqdm import tqdm
from dataset_wrapper import VideoDatasetWrapper
from utils import calculate_metrics_pixel_space_frame_2_frame, calculate_metrics_latent_space, visualize_frames, \
    z_score_normalization, unit_norm_normalization


class Evaluator:
    def __init__(self, name, dataset: VideoDatasetWrapper, frame_skip: int = 0, repo_id="stabilityai/stable-video-diffusion-img2vid",
                 local_dir: str = None, image_vae: bool=True, use_local_dir: bool = False,):
        self.name = name
        self.dataset = dataset
        self.frame_skip = frame_skip
        self.local_dir = local_dir
        self.image_vae = image_vae
        self.use_local_dir = use_local_dir
        if self.use_local_dir or local_dir is not None:
            self.vae = AutoencoderKL.from_pretrained(self.local_dir)
        else:
            try:
                self.pipe = DiffusionPipeline.from_pretrained(repo_id)
                self.vae = self.pipe.vae
            except:
                self.vae = AutoencoderKL.from_pretrained(repo_id)

        print(f"The Evaluator Pipeline for {self.name} is initialized")
        print(f"Initializing {self.dataset.name} dataset with frame skip {self.frame_skip}...")
        self.test_dataloader = self.dataset.test_loader
        print(f"The {self.dataset.name} dataset object is created")

    def eval_vae(self):
        class_ind_path = "datasets/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
        # Prepare label indices
        class_mapping = self.dataset.parse_class_indices(class_ind_path)
        print(class_mapping)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)
        self.vae.eval()

        # Initialize metrics storage
        metrics_by_label = {}

        # Loop through the dataset
        for batch_idx, (video_clips, labels) in enumerate(tqdm(self.test_dataloader)):
            # Video clip shape: (Batch, Frame, Channel, Height, Weight)
            video_clips = video_clips.to(device)
            labels = labels.to(device)

            for i in range(video_clips.size(0)):  # Loop through batch
                latent_metrics = None
                reconstruction_metrics = None

                clip = video_clips[i]  # Single video clip
                label = str(labels[i].item() + 1)  # Numeric label to string
                textual_label = class_mapping[label]  # Textual label from mapping

                if self.image_vae:
                    reconstructed_frames = []
                    latent_vectors = []
                    #for frame in clip:
                    for frame_idx in range(0, len(clip), self.frame_skip):
                        frame = clip[frame_idx]
                        frame = frame.unsqueeze(0)  # Add batch dimension
                        # Encode and decode the clip
                        with torch.no_grad():
                            latent = self.vae.encode(frame).latent_dist.sample()
                            reconstruction = self.vae.decode(latent).sample

                            # visualize_frames(frame, reconstruction)

                            reconstructed_frames.append(reconstruction)
                            latent_vectors.append(latent)
                    # latent_vectors = unit_norm_normalization(latent_vectors)
                    latent_metrics = calculate_metrics_latent_space(latent_vectors)
                    reconstruction_metrics = calculate_metrics_pixel_space_frame_2_frame(reconstructed_frames)
                else:
                    # Video Eval
                    latent_vectors = self.vae.encode(clip).latent_dist.sample()
                    reconstructed_frames = self.vae.decode(latent_vectors, num_frames=clip.size(0)).sample

                    latent_metrics = calculate_metrics_latent_space(latent_vectors)
                    reconstruction_metrics = calculate_metrics_pixel_space_frame_2_frame(reconstructed_frames)

                # Aggregate metrics for the label
                if textual_label not in metrics_by_label:
                    metrics_by_label[textual_label] = {
                        "Euclidean Distance": [],
                        "Cosine Similarity": [],
                        "PSNR": [],
                        "SSIM": []
                    }
                metrics_by_label[textual_label]["Euclidean Distance"].append(latent_metrics[0])
                metrics_by_label[textual_label]["Cosine Similarity"].append(latent_metrics[1])
                metrics_by_label[textual_label]["PSNR"].append(reconstruction_metrics[0])
                metrics_by_label[textual_label]["SSIM"].append(reconstruction_metrics[1])

            # Optional: Stop after evaluating a few batches for testing
            # if batch_idx >= 1:
            #     break

        self.save_data(metrics_by_label)

    def save_data(self, metrics_by_label):
        # Prepare data for the DataFrame
        data = {}
        for textual_label, metrics in metrics_by_label.items():
            data[textual_label] = {metric: sum(values) / len(values) for metric, values in metrics.items()}

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')

        print(df)

        df.to_csv("./results/"+self.name+"-"+self.dataset.name+".csv")