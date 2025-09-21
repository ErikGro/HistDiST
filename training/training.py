#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
import statistics
import random

import accelerate
import datasets
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import timm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoImageProcessor, ViTModel
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.transforms import v2
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
from torchvision import transforms

from datasets import Dataset, Features
from datasets import Image as ImageFeature

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.1")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--translation_prompt",
        type=str,
        default="IHC",
        help="The prompt used as text conditioning for H&E->IHC translation, will be the same for all batches",
    )
    parser.add_argument(
        "--data_set_dir",
        type=str,
        help="The path to the dir to load pairs of H&E and IHC images from",
    )
    parser.add_argument(
        "--he_generation_prompt",
        type=str,
        default="H&E",
        help="The prompt used as text conditioning H&E generation, will be the same for all batches",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--bias_he_ihc",
        type=float,
        default=0.5,
        help="Blends between exclusive H&E generation at 0 and exclusive H&E->IHC translation at 1"
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    def gen_examples(path, num_samples_to_use):
        def fn():
            for he_path in list(Path(path + "/TrainValAB/trainA").glob('*.jpg'))[:num_samples_to_use]:
                yield {
                    "he_image": {"path": str(he_path)},
                    "ihc_image": {"path": str(he_path).replace("trainA", "trainB")}
                }

        return fn


    dataset = Dataset.from_generator(
        gen_examples(args.data_set_dir, num_samples_to_use=4096),
        features=Features(
            he_image=ImageFeature(),
            ihc_image=ImageFeature()
        ),
    )
    
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    model.to(accelerator.device)
    
    # Logging 
    num_inference_steps = 20
    image_guidance_scale = 0
    guidance_scale = 0
    inference_batch_size = 4
    dir_he = Path("/graphics/scratch2/students/grosskop/benchmark_er_testset/valA/")
    dir_ihc_target = Path("/graphics/scratch2/students/grosskop/benchmark_er_testset/valB")
    dir_ihc_genereated = os.path.join(args.output_dir, "generated/ihc")
    dir_he_genereated = os.path.join(args.output_dir, "generated/he")
    os.makedirs(dir_ihc_genereated, exist_ok=True)
    os.makedirs(dir_he_genereated, exist_ok=True)


    def convert_to_np(image):
        image = image.convert("RGB")
        return np.array(image).transpose(2, 0, 1)
        
        
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

    def genreateImages(pipe):
        steps = 1000 // inference_batch_size
        for step in tqdm(range(steps), total=steps):
            indices = list(range(step * inference_batch_size, (step + 1) * inference_batch_size))
            batch = list(map(lambda l: PIL.Image.open(f"{dir_he}/{l:03d}.jpg"), indices))
            tensor_batch = torch.stack([v2.ToTensor()(image) for image in batch])

            # inputs = image_processor(tensor_batch, return_tensors="pt", do_rescale=False).to(accelerator.device)
            # with torch.no_grad():
                # hidden_states = model(**inputs).last_hidden_state.to(accelerator.device)

            ihc_generated = pipe([args.translation_prompt] * len(tensor_batch),
                image=tensor_batch,
                prompt_embeds=hidden_states,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
            ).images
            
            he_generated = pipe([args.he_generation_prompt] * len(tensor_batch),
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
            ).images

            for batch_index, image in enumerate(ihc_generated):
                file_index = step * inference_batch_size + batch_index
                image.save(f"{dir_ihc_genereated}/{file_index:03d}.jpg")
            
            for batch_index, image in enumerate(he_generated):
                file_index = step * inference_batch_size + batch_index
                image.save(f"{dir_he_genereated}/{file_index:03d}.jpg")


    def computeMetrics():
        ssim_scores = []
        psnr_scores = []
        for index, path in tqdm(enumerate(dir_ihc_target.glob('*.jpg')), total=1000):
            groundtruth = PIL.Image.open(path).convert("RGB")
            generated = PIL.Image.open(str(os.path.join(dir_ihc_genereated, path.name))).convert("RGB")

            ihcGeneratedCV = cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2GRAY)
            ihcGroundTruthCV = cv2.cvtColor(np.asarray(groundtruth), cv2.COLOR_RGB2GRAY)

            ssimValue = ssim(ihcGroundTruthCV, ihcGeneratedCV)
            psnrValue = cv2.PSNR(ihcGroundTruthCV, ihcGeneratedCV)

            psnr_scores.append(psnrValue)
            ssim_scores.append(ssimValue)
            
        fid_ihc = compute_fid(dir_ihc_target, dir_ihc_genereated)
        fid_he = compute_fid(dir_he, dir_he_genereated)
        
        return statistics.mean(ssim_scores), statistics.mean(psnr_scores), fid_ihc, fid_he


    def log_validation(
        pipeline,
        args,
        accelerator
        ):
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.translation_prompt}."
        )
        
        if args.prediction_type == "v_prediction":
            pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
        elif args.prediction_type == "epsilon":
            pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        else:
            raise ValueError("args.prediction_type has to be either epsilon or v_prediction")
    
        pipeline.set_progress_bar_config(disable=True)
        
        autocast_ctx = torch.autocast(accelerator.device.type)

        translated_images = []
        he_images = []
        with autocast_ctx:
            genreateImages(pipeline)
            ssim_score, psnr_score, fid_ihc, fid_he = computeMetrics()
            
            he_image = PIL.Image.open("val_image_he.jpg")     
            tensor_batch = torch.stack([v2.ToTensor()(image) for image in [he_image]])
            # inputs = image_processor(tensor_batch, return_tensors="pt", do_rescale=False).to(accelerator.device)
            # with torch.no_grad():
            #     hidden_states = model(**inputs).last_hidden_state.to(accelerator.device)
                                
            for i in range(args.num_validation_images):
                translated_images.append(
                    pipeline(
                        # args.translation_prompt,
                        image=he_image,
                        prompt_embeds=hidden_states,
                        num_inference_steps=num_inference_steps,
                        image_guidance_scale=image_guidance_scale,
                        guidance_scale=guidance_scale,
                        # generator=torch.Generator(device=accelerator.device).manual_seed(i),
                    ).images[0]
                )
                
                he_images.append(
                    pipeline(
                        args.he_generation_prompt,
                        num_inference_steps=num_inference_steps,
                        image_guidance_scale=0,
                        guidance_scale=0,
                        # generator=torch.Generator(device=accelerator.device).manual_seed(i),
                    ).images[0]
                )
                
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation_ihc": [
                            wandb.Image(image, caption="002.jpg pred")
                            for i, image in enumerate(translated_images)
                        ],
                        "validation_he": [
                            wandb.Image(image, caption="HE unconditional")
                            for i, image in enumerate(he_images)
                        ],
                        "ssim": ssim_score, 
                        "psnr": psnr_score, 
                        "fid ihc": fid_ihc,
                        "fid he": fid_he,
                    }
                )
                
            del pipeline
            torch.cuda.empty_cache()


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    if args.prediction_type == "v_prediction":
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
    elif args.prediction_type == "epsilon":
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        raise ValueError("args.prediction_type has to be either epsilon or v_prediction")
        
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias[:len(unet.conv_in.bias)].copy_(unet.conv_in.bias)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, angle=random.choice([0, 90, 180, 270])))
        ]
    )

    def preprocess_images(examples):
        he_images = np.concatenate(
            [convert_to_np(image) for image in examples["he_image"]]
        )
        ihc_images = np.concatenate(
            [convert_to_np(image) for image in examples["ihc_image"]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([he_images, ihc_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        he_images, ihc_images = preprocessed_images.chunk(2)
        he_images = he_images.reshape(-1, 3, args.resolution, args.resolution)
        ihc_images = ihc_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["he_images"] = he_images
        examples["ihc_images"] = ihc_images

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)
        
        # tokenize translation prompt once, which is the same for all batches
        translation_prompt = tokenize_captions([args.translation_prompt] * args.train_batch_size)
        translation_prompt = text_encoder(translation_prompt)[0].to(accelerator.device)
        
        he_prompt = tokenize_captions([args.he_generation_prompt] * args.train_batch_size)
        he_prompt = text_encoder(he_prompt)[0].to(accelerator.device)
        

    def collate_fn(examples):
        he_pixel_values = torch.stack([example["he_images"] for example in examples])
        he_pixel_values = he_pixel_values.to(memory_format=torch.contiguous_format).float()
        ihc_pixel_values = torch.stack([example["ihc_images"] for example in examples])
        ihc_pixel_values = ihc_pixel_values.to(memory_format=torch.contiguous_format).float()
        return {
            "he_pixel_values": he_pixel_values,
            "ihc_pixel_values": ihc_pixel_values
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project, config=vars(args), init_kwargs={"wandb": {"name": args.name, "notes": args.description}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        epoch_mse_losses = []
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                generate_he = step % 2 == 0
                # generate_he = False # ONLY TRANSLATE
                # mask_he = (torch.rand(len(batch["ihc_pixel_values"])) > args.bias_he_ihc).bool()

                batch_pixels = batch["he_pixel_values" if generate_he else "ihc_pixel_values"]
                # batch_pixels[mask_he] = batch["he_pixel_values"][mask_he]
                target_latents = vae.encode(batch_pixels.to(weight_dtype)).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                noise = torch.randn_like(target_latents)
                noise += args.noise_offset * torch.randn(target_latents.shape[0], target_latents.shape[1], 1, 1).to(accelerator.device) # Add offset noise
                noise += args.input_perturbation * torch.randn_like(noise) # Input perturbation
                
                bsz = target_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                he_image_embeds = vae.encode(batch["he_pixel_values"].to(weight_dtype)).latent_dist.mode()
                # he_image_embeds[mask_he] = torch.zeros_like(he_image_embeds)[mask_he]
                if generate_he:
                    he_image_embeds = torch.zeros_like(he_image_embeds)
                
                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, he_image_embeds], dim=1)
                    
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # prompt_embeds = translation_prompt
                # prompt_embeds[mask_he] = he_prompt[mask_he]
                if generate_he:
                    hidden_states = he_prompt
                else:
                    # hidden_states = translation_prompt
                    embeds_input = (batch["he_pixel_values"] + 1) / 2
                    inputs = image_processor(embeds_input, return_tensors="pt", do_rescale=False).to(target_latents.device)
                    with torch.no_grad():
                        hidden_states = model(**inputs).last_hidden_state.to(target_latents.device)

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, hidden_states, return_dict=False)[0]
                
                if args.snr_gamma is None:
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape)))) * mse_loss_weights
                    mse_loss = mse_loss.mean()
                    
                epoch_mse_losses.append(mse_loss.item())
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(mse_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        print(f"EPOCH {epoch} finished")
        if accelerator.is_main_process:
            accelerator.log({"train_loss_epoch": statistics.mean(epoch_mse_losses)}, step=global_step)

            if (epoch % args.validation_epochs == 0 and epoch != (args.num_train_epochs - 1)):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    safety_checker = None,
                    requires_safety_checker = False
                )

                log_validation(
                    pipeline,
                    args,
                    accelerator
                )

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()


    print(f"Training finished")
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
            safety_checker = None,
            requires_safety_checker = False
        )
        pipeline.save_pretrained(args.output_dir)

        log_validation(
            pipeline,
            args,
            accelerator
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
