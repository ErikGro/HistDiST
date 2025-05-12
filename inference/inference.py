import PIL
import torch
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline, DDIMInverseScheduler
import math 
from torchvision.transforms import v2
from transformers import AutoImageProcessor, ViTModel
import argparse

def main():
    args = parse_args()

    ############ Eta schedules
    eta_zero = lambda x: 0
    eta_constant = lambda x: 0.1
    eta_linear = lambda x: x
    eta_cosine_min_10 = lambda x: 0.55 - (0.45 / 1) * math.cos(math.pi * x)
    eta_cosine_min_20 = lambda x: 0.6 - (0.4 / 1) * math.cos(math.pi * x)
    eta_cosine_min_30 = lambda x: 0.65 - (0.35 / 1) * math.cos(math.pi * x)

    ########## Inversion and translation
    input_pil_image = PIL.Image.open(args.img_path)
    input_batch = torch.stack([v2.ToTensor()(image) for image in [input_pil_image]])
    translated = inversion_translate(args, input_batch, eta_zero, eta_cosine_min_20)
    translated[0].save(f"output.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--prediction_type', type=str, choices=["v_prediction", "epsilon"], default="v_prediction")
    parser.add_argument('--steps_inversion', type=int, default=100)
    parser.add_argument('--steps_translation', type=int, default=100)
    parser.add_argument('--normalize', type=str, choices=["steps", "inversion", "none"], default="inversion")
    parser.add_argument('--prompt_inversion', type=str, default="H&E")
    parser.add_argument('--prompt_translation', type=str, default="IHC")
    parser.add_argument('--dtype', type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument('--device', type=str, default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def inversion_translate(args, input_batch, eta_schedule_inversion, eta_schedule_translation):
    dtype = getattr(torch, args.dtype)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_folder_path).to(device=args.device,    
                                                                                             dtype=dtype)
    batch_size = input_batch.shape[0]
    input_batch_encoded = pipe.vae.encode(input_batch.to(dtype=dtype, device=args.device) * 2 - 1)
    he_image_embeds = input_batch_encoded.latent_dist.mode()
    encoder_hidden_states_inversion = pipe._encode_prompt(
        args.prompt_inversion, args.device, batch_size, False
    )
    encoder_hidden_states_translation = pipe._encode_prompt(
        args.prompt_translation, args.device, batch_size, False
    )

    feature_extractor = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        
    ###### Inversion
    if args.prediction_type == "v_prediction":
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(args.model_folder_path, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
    elif args.prediction_type == "epsilon":
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(args.model_folder_path, subfolder="scheduler", prediction_type="epsilon")
    else:
        raise ValueError("prediction_type has to be either epsilon or v_prediction")
        
    pipe.scheduler.set_timesteps(args.steps_inversion, device=args.device)
    input_batch_inversion = 0.18215 * input_batch_encoded.latent_dist.sample()
    latents_inversion = input_batch_inversion.clone().to(args.device)

    print("Performing inversion")
    for i in tqdm(range(0, args.steps_inversion), total=args.steps_inversion):
        x = float(i) / float(args.steps_inversion - 1)
        
        timestep_current = pipe.scheduler.timesteps[i]
        latents_inversion_scaled = pipe.scheduler.scale_model_input(latents_inversion, timestep_current)
        latent_model_input = torch.cat([latents_inversion_scaled, torch.zeros_like(latents_inversion_scaled)], dim=1)
        
        model_output = pipe.unet(latent_model_input, timestep_current, encoder_hidden_states=encoder_hidden_states_inversion).sample
                
        timestep_next = min(
            timestep_current - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, pipe.scheduler.config.num_train_timesteps - 1
        )

        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[timestep_next] if timestep_next >= 0 else pipe.scheduler.initial_alpha_cumprod
        alpha_prod_t_current = pipe.scheduler.alphas_cumprod[timestep_current]
        beta_prod_t_next = 1 - alpha_prod_t_next
        beta_prod_t_current = 1 - alpha_prod_t_current
        
        if pipe.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (latents_inversion - beta_prod_t_next ** (0.5) * model_output) / alpha_prod_t_next ** (0.5)
            pred_epsilon = model_output
        elif pipe.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t_next**0.5) * latents_inversion - (beta_prod_t_next**0.5) * model_output
            pred_epsilon = (alpha_prod_t_next**0.5) * model_output + (beta_prod_t_next**0.5) * latents_inversion
            
        eta = max(min(eta_schedule_inversion(x), 1), 0)

        variance = (beta_prod_t_next / beta_prod_t_current) * (1 - alpha_prod_t_current / alpha_prod_t_next) # option A
        std_dev_t = eta * variance ** (0.5)
        
        if eta > 0:
            pred_sample_direction = (1 - alpha_prod_t_current - std_dev_t**2) ** (0.5) * pred_epsilon
        else:
            pred_sample_direction = (1 - alpha_prod_t_current) ** (0.5) * pred_epsilon
        
        prev_sample = alpha_prod_t_current ** (0.5) * pred_original_sample + pred_sample_direction

        if args.normalize == "steps":
            prev_sample = (prev_sample - prev_sample.mean(dim=(2,3), keepdim=True)) / prev_sample.std(dim=(2,3), keepdim=True)
            
        if eta > 0:
            prev_sample = prev_sample + std_dev_t * torch.randn_like(latents_inversion)
            
        latents_inversion = prev_sample

        
    ##### Normalize final latents
    if args.normalize == "inversion":
        latents_inversion = (latents_inversion - latents_inversion.mean(dim=(2,3), keepdim=True)) / latents_inversion.std(dim=(2,3), keepdim=True)


    ##### Features extractor for translation
    feature_extractor.to(pipe.device)
    inputs = image_processor(input_batch, return_tensors="pt", do_rescale=False).to(device=pipe.device, dtype=dtype)
    encoder_hidden_states_translation = feature_extractor(**inputs).last_hidden_state.to(device=pipe.device, dtype=dtype)


    ###### translation
    if args.prediction_type == "v_prediction":
        pipe.scheduler = DDIMScheduler.from_pretrained(args.model_folder_path, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
    elif args.prediction_type == "epsilon":
        pipe.scheduler = DDIMScheduler.from_pretrained(args.model_folder_path, subfolder="scheduler", prediction_type="epsilon")
    else:
        raise ValueError("prediction_type has to be either epsilon or v_prediction")

    pipe.scheduler.set_timesteps(args.steps_translation, device=args.device)

    latents_translation = latents_inversion.clone()
    print("Performing translation")    
    for i in tqdm(range(0, args.steps_translation), total=args.steps_translation):
        timestep_next = pipe.scheduler.timesteps[i]
        latent_model_input = pipe.scheduler.scale_model_input(latents_translation, timestep_next)
        latent_model_input = torch.cat([latent_model_input, he_image_embeds], dim=1)
        noise_pred = pipe.unet(latent_model_input, timestep_next, encoder_hidden_states=encoder_hidden_states_translation).sample
        x = float(i) / float(args.steps_translation - 1)
        eta = max(min(eta_schedule_translation(x), 1), 0)
        latents_translation = pipe.scheduler.step(noise_pred, timestep_next, latents_translation, eta=eta).prev_sample
            
    return pipe.numpy_to_pil(pipe.decode_latents(latents_translation))


if __name__ == "__main__":
    main()
    