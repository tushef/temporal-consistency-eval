
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Lambda

from dataset_wrapper import VideoDatasetWrapper
from dcae_eval import DCAEEvaluator
from vae_eval import Evaluator


# Models to fetch
MODELS = {
    "SD-1.5": "runwayml/stable-diffusion-v1-5",
    "vae": "stabilityai/sd-vae-ft-mse",
    "sdxl": "stabilityai/sdxl-vae",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "open-sora" : "hpcai-tech/OpenSora-VAE-v1.2"
}

def main():
    transform = Compose([
        # scale in [0, 1]
        Lambda(lambda x: x / 255.),
        # reshape into (T, C, H, W)
        Lambda(lambda x: x.permute(0, 3, 1, 2))
    ])

    FRAME_SKIP = 1
    FRAMES_PER_CLIP = 24
    NUM_CLIPS_PER_LABEL = int(30 / int(FRAMES_PER_CLIP / FRAME_SKIP))
    print("PARAMS:")
    print(FRAME_SKIP, FRAMES_PER_CLIP, NUM_CLIPS_PER_LABEL)

    dataset = VideoDatasetWrapper(name="UCF101",
                                    dataset=UCF101,
                                    frames_per_clip=FRAMES_PER_CLIP,
                                    step_between_clips=FRAMES_PER_CLIP, # get only unique clips
                                    num_clips_per_label= NUM_CLIPS_PER_LABEL,
                                    train=False,
                                    transform=transform,
                                    collate_callback=None,
                                    )
    evaluator = Evaluator("SD15-VAE-EMA-FrameSkip_"+str(FRAME_SKIP), dataset,
                          repo_id="runwayml/stable-diffusion-v1-5",
                          image_vae=True,
                          frame_skip=FRAME_SKIP)
    evaluator.eval_vae()
    evaluator = Evaluator("FLUX-VAE_FrameSkip" + str(FRAME_SKIP), dataset,
                          local_dir="models\\checkpoints\\flux",
                          frame_skip=FRAME_SKIP)
    evaluator.eval_vae()
    evaluator = DCAEEvaluator("DCAE-FrameSkip_"+str(FRAME_SKIP), dataset, frame_skip=FRAME_SKIP)
    evaluator.eval_dcae()
    # evaluator = Evaluator("Stable Video Diffusion Temporal-VAE", dataset,
    #                       repo_id="stabilityai/stable-video-diffusion-img2vid",
    #                       image_vae=False)
    # evaluator.eval_vae()

    #Load the saved Stable Diffusion model
    # pipe = StableDiffusionPipeline.from_pretrained(local_model_dir)
    # Load the saved VAE model
    #vae = AutoencoderKL.from_pretrained("./models/checkpoints/sd-vae-ft-ema")
    #vae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0")
    # # Assign the VAE to the pipeline
    # pipe.vae = vae


if __name__ == '__main__':
    main()