import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda():
    # Load data
    flux = pd.read_csv('./results/Flux VAE-UCF101.csv')
    dcae = pd.read_csv('./results/Deep Compression Autoencoder-UCF101.csv')
    sd = pd.read_csv('./results/Stable Diffusion VAE-EMA-UCF101.csv')
    dcae_2 = pd.read_csv('./results/DCAE-FrameSkip_2-UCF101.csv')
    sd_2 = pd.read_csv('./results/SD15-VAE-EMA-Normalized-FrameSkip_2-UCF101.csv')
    dcae_5 = pd.read_csv('./results/DCAE-FrameSkip_5-UCF101.csv')
    sd_5 = pd.read_csv('./results/SD15-VAE-EMA-Normalized-FrameSkip_5-UCF101.csv')
    dcae_10 = pd.read_csv('./results/DCAE-FrameSkip_10-UCF101.csv')
    sd_10 = pd.read_csv('./results/SD15-VAE-EMA-Normalized-FrameSkip_10-UCF101.csv')

    numerical_columns = ['Euclidean Distance', 'Cosine Similarity', 'PSNR', 'SSIM']

    mean_sd_1 = sd[numerical_columns].mean()
    mean_sd_2 = sd_2[numerical_columns].mean()
    mean_sd_5 = sd_5[numerical_columns].mean()
    mean_sd_10 = sd_10[numerical_columns].mean()

    mean_dcae_1 = dcae[numerical_columns].mean()
    mean_dcae_2 = dcae_2[numerical_columns].mean()
    mean_dcae_5 = dcae_5[numerical_columns].mean()
    mean_dcae_10 = dcae_10[numerical_columns].mean()

    for column in numerical_columns:
        points_dist = [(1, mean_sd_1[column]),
                    (2, mean_sd_2[column]),
                                 (5, mean_sd_5[column]),
                                 (10, mean_sd_10[column])]
        points_dcae_dist = [(1, mean_dcae_1[column]),(2, mean_dcae_2[column]),
                                      (5, mean_dcae_5[column]),
                                      (10, mean_dcae_10[column])]

        x, y = zip(*points_dist)
        plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=8, label="SD-VAE")
        x, y = zip(*points_dcae_dist)
        plt.plot(x, y, marker='o', linestyle='-', color='r', markersize=8, label="DC-AE")

        plt.xlabel("Number of Frames Skipped")
        plt.ylabel("Mean " + column)
        plt.title("Mean " + column + " vs Number of Frames Skipped")
        plt.legend()

        plt.show()

    # Rename the first column for clarity if necessary
    # data.rename(columns={data.columns[0]: 'Activity'}, inplace=True)

    # Summary statistics
    print("Summary statistics:\n")
    print(sd.describe())
    print(flux.describe())
    print(dcae.describe())

    for column in numerical_columns:
        plt.figure(figsize=(8, 5))
        #sns.histplot(flux[column], kde=True, bins=30, color='green', alpha=0.5, label="FLUX-VAE")
        #sns.histplot(sd[column], kde=True, bins=30, color='blue', alpha=0.5, label="SD-VAE")
        sns.histplot(dcae[column], kde=True, bins=30, color='red', alpha=0.5, label="DC-AE")
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    eda()