from dataclasses import dataclass
import os
import random
import torch
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor 
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from typing import List
from util.perceptual import PerceptualHashValue
import timm
from PIL import Image
from torchvision import transforms
import torch
from tqdm.auto import tqdm
import PIL
import PIL.Image
import os
from torch import nn
import time
import argparse


def main():
    args = parse_args()
    metrics = compute_metrics(args.target_dir, args.generated_dir)
    print(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--generated_dir', type=str, required=True)
    
    return parser.parse_args()


@dataclass
class Metrics:
    ssim: float
    psnr: float
    fid: float
    phv: List[float]
    acc: float
    accTop3: float
    
    
def feature_file_name(dir):
    return f"features_{os.path.basename(dir)}.pt"   


def compute_metrics(target_dir: str, generated_dir: str, device = "cuda") -> Metrics:
    start = time.time()

    img_list = [f for f in os.listdir(generated_dir) if f.endswith(('png', 'jpg'))]
    img_format = '.' + img_list[0].split('.')[-1]
    img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
    random.seed(0)
    random.shuffle(img_list)
    
    print("Extract features for accuracy")
    compute_features(generated_dir)
    compute_features(target_dir)
    acc, accTop3 = compute_accuracy(feature_file_name(target_dir), feature_file_name(generated_dir), "cosine")
    
    # PHV statistics
    print("Compute PHV")
    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
    PHV = PerceptualHashValue(
            T=0.01, network='resnet50', layers=layers, 
            resize=False, resize_mode='bilinear',
            instance_normalized=False)
    PHV.to(device)
    all_phv = []
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(generated_dir, i + img_format))
        real = io.imread(os.path.join(target_dir, i + img_format))

        fake = to_tensor(fake).to(device)
        real = to_tensor(real).to(device)

        phv_list = PHV(fake, real)
        all_phv.append(phv_list)
    all_phv = np.array(all_phv)
    all_phv = np.mean(all_phv, axis=0).tolist()

    # FID statistics
    print("Compute FID")
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    real_paths = [os.path.join(target_dir, f + ".jpg") for f in img_list]
    fake_paths = [os.path.join(generated_dir, f + img_format) for f in img_list]

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)


    m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)


    # PSNR and SSIM statistics
    print("Compute PSNR/SSIM")
    psnr = []
    ssim = []
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(generated_dir, i + img_format))
        real = io.imread(os.path.join(target_dir, i + ".jpg"))
        PSNR = peak_signal_noise_ratio(fake, real)
        psnr.append(PSNR)
        SSIM = structural_similarity(fake, real, multichannel=True, channel_axis=2)
        ssim.append(SSIM)
    average_psnr = sum(psnr)/len(psnr)
    average_ssim = sum(ssim)/len(ssim)
    
    print("Compute KID")
    command = f'python3 util/kid_score.py --true {target_dir} --fake {generated_dir}'
    os.system(command)
    
    end = time.time()
    print("Finished after: ", (end - start))

    return Metrics(average_ssim, average_psnr, fid_value, all_phv, acc, accTop3)



# Login required, access restricted
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to("cuda")
tile_encoder.eval()


def compute_features(dir, device = "cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    batch_size = 50
    features = []
    with torch.no_grad():
        steps_total = 1000 // batch_size
        for step in tqdm(range(steps_total), total=steps_total):    
                indices = list(range(step * batch_size, (step + 1) * batch_size))
                batch = list(map(lambda l: PIL.Image.open(f"{dir}/{l:03d}.jpg"), indices))
                tensor_batch = torch.stack([transform(image) for image in batch]).to(device)
                features.append(tile_encoder(tensor_batch))
            
    torch.save(torch.cat(features, dim=0), feature_file_name(dir))
    

cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

def compute_accuracy(file_groundtruth, file_prediction, distance_type):
    features_ihc_ground_truth = torch.load(file_groundtruth).cpu()
    features_pred = torch.load(file_prediction).cpu()
    
    index_distances = []
    for i, pred in tqdm(enumerate(features_pred), total=len(features_pred)):
        distances = []
        
        for j, groundtruth in enumerate(features_ihc_ground_truth):
            if distance_type == "euclidean":
                distance = torch.dist(features_pred[i], features_ihc_ground_truth[j], p=2)
            elif distance_type == "cosine":
                distance = cosine_sim(features_pred[i], features_ihc_ground_truth[j])
            
            distances.append(distance)
            
            if i == j:
                pair_distance = distance

        sorted_distances = torch.sort(torch.tensor(distances), descending=distance_type == "cosine")[0]    
        absolute_distance = torch.where(sorted_distances == pair_distance)[0][0].item() #(sorted_distances == pair_distance).nonzero(as_tuple=True)[0] # Get index from sorted distances
        index_distances.append(absolute_distance)
   
        
    accuracy = float(sum(index == 0 for index in index_distances)) / len(features_pred)
    accuracy_top_3 = float(sum(index < 3 for index in index_distances)) / len(features_pred)
    
    return accuracy, accuracy_top_3
    
    
def compute_fid(targ_dir: str, pred_dir: str, device="cuda") -> float:
    img_list = [f for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))]
    img_format = '.' + img_list[0].split('.')[-1]
    img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
    random.seed(0)
    random.shuffle(img_list)
    
    # FID statistics
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    real_paths = [os.path.join(targ_dir, f + img_format) for f in img_list]
    fake_paths = [os.path.join(pred_dir, f + img_format) for f in img_list]
    print(f"Total number of images: {len(real_paths)}")

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value
    
    
if __name__ == "__main__":
    main()
