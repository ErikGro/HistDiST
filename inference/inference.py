import PIL
import torch
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline, DDIMInverseScheduler
import os 
import math 
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ViTModel, Swinv2Model

feature_extractor = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")

@torch.no_grad()
def inversion_translate(pipe, prediction_type, device, input_batch, prompt_inversion, prompt_translation, steps_inversion, steps_translation, eta_schedule_inversion, eta_schedule_translation, normalize):
    batch_size = input_batch.shape[0]
    input_batch_encoded = pipe.vae.encode(input_batch.to(device) * 2 - 1)
    he_image_embeds = input_batch_encoded.latent_dist.mode()
    encoder_hidden_states_inversion = pipe._encode_prompt(
        prompt_inversion, device, batch_size, False
    )
    encoder_hidden_states_translation = pipe._encode_prompt(
        prompt_translation, device, batch_size, False
    )
        
    ###### Inversion
    if prediction_type == "v_prediction":
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
    elif prediction_type == "epsilon":
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler", prediction_type="epsilon")
    else:
        raise ValueError("prediction_type has to be either epsilon or v_prediction")
        
    pipe.scheduler.set_timesteps(steps_inversion, device=device)
    input_batch_inversion = 0.18215 * input_batch_encoded.latent_dist.sample()
    latents_inversion = input_batch_inversion.clone().to(device)
    
    for i in range(0, steps_inversion):
        x = float(i) / float(steps_inversion - 1)
        
        timestep_current = pipe.scheduler.timesteps[i]
        latents_inversion_scaled = pipe.scheduler.scale_model_input(latents_inversion, timestep_current)
        latent_model_input = torch.cat([latents_inversion_scaled, torch.zeros_like(latents_inversion_scaled)], dim=1)
        
        encoder_hidden_states = encoder_hidden_states_inversion
        model_output = pipe.unet(latent_model_input, timestep_current, encoder_hidden_states=encoder_hidden_states).sample
                
        timestep_next = min(
            timestep_current - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, pipe.scheduler.config.num_train_timesteps - 1
        )

        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[timestep_next] if timestep_next >= 0 else pipe.scheduler.initial_alpha_cumprod
        alpha_prod_t_current = pipe.scheduler.alphas_cumprod[timestep_current]
        alphas.append(alpha_prod_t_current)
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

        if normalize == "steps":
            prev_sample = (prev_sample - prev_sample.mean(dim=(2,3), keepdim=True)) / prev_sample.std(dim=(2,3), keepdim=True)
            
        if eta > 0:
            prev_sample = prev_sample + std_dev_t * torch.randn_like(latents_inversion)
            
        latents_inversion = prev_sample

        
    ##### Normalize final latents
    if normalize == "inversion":
        latents_inversion = (latents_inversion - latents_inversion.mean(dim=(2,3), keepdim=True)) / latents_inversion.std(dim=(2,3), keepdim=True)


    ##### Features extractor for translation
    feature_extractor.to(pipe.device)
    inputs = image_processor(input_batch, return_tensors="pt", do_rescale=False).to(pipe.device)
    encoder_hidden_states_translation = feature_extractor(**inputs).last_hidden_state.to(pipe.device)
    

    ###### translation
    if prediction_type == "v_prediction":
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", prediction_type="v_prediction", timestep_spacing="trailing", rescale_betas_zero_snr=True)
    elif prediction_type == "epsilon":
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", prediction_type="epsilon")
    else:
        raise ValueError("prediction_type has to be either epsilon or v_prediction")
    
    pipe.scheduler.set_timesteps(steps_translation, device=device)

    latents_translation = latents_inversion.clone()        
    for i in range(0, steps_translation):
        timestep_next = pipe.scheduler.timesteps[i]
        latent_model_input = pipe.scheduler.scale_model_input(latents_translation, timestep_next)
        latent_model_input = torch.cat([latent_model_input, he_image_embeds], dim=1)
        noise_pred = pipe.unet(latent_model_input, timestep_next, encoder_hidden_states=encoder_hidden_states_translation).sample
        alphas.append(pipe.scheduler.alphas_cumprod[timestep_next])
        x = float(i) / float(steps_translation - 1)
        eta = max(min(eta_schedule_translation(x), 1), 0)
        latents_translation = pipe.scheduler.step(noise_pred, timestep_next, latents_translation, eta=eta).prev_sample
            
    return pipe.numpy_to_pil(pipe.decode_latents(latents_translation))


############ Config general
device = "cuda:0"
model_id = "path/to/model/dir"
prediction_type = "v_prediction"
dType = torch.bfloat16
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id).to(device=device, dType=dType)

############ Eta schedules
eta_zero = lambda x: 0
eta_constant = lambda x: 0.1
eta_linear = lambda x: x
eta_cosine_min_10 = lambda x: 0.55 - (0.45 / 1) * math.cos(math.pi * x)
eta_cosine_min_20 = lambda x: 0.6 - (0.4 / 1) * math.cos(math.pi * x)
eta_cosine_min_30 = lambda x: 0.65 - (0.35 / 1) * math.cos(math.pi * x)

############ Inversion parameters
prompt_inversion = "H&E"
prompt_translation = "IHC"
steps_inversion = 100
steps_translation = 100
eta_schedule_inversion = eta_zero
eta_schedule_translation = eta_zero
normalize = None # "steps", "inversion", None

########## Inversion and translation
input_er_image = PIL.Image.open("./example_images/er.jpg")
input_batch = torch.stack([v2.ToTensor()(image) for image in [input_er_image]])
translated = inversion_translate(pipe, prediction_type, device, input_batch, prompt_inversion, prompt_translation, steps_inversion, steps_translation, eta_schedule_inversion, eta_schedule_translation, normalize)
translated[0].save(f"output.jpg")
