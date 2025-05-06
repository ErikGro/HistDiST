MODEL_NAME="botp/stable-diffusion-v1-5"
PROJECT_NAME="Thesis InstructPix2Pix"
RUN_TITLE="25_02_26 Evaluation PR joint VPred gamma 5 300 xattention"
RUN_DESCRIPTION="baseline"
OUTPUT_DIR="${RUN_TITLE// /_}"

export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

nohup accelerate launch --num_processes=3 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no --gpu_ids $CUDA_VISIBLE_DEVICES train_instruct_pix2pix_jointly.py \
    --num_train_epochs=300 \
    --validation_epochs=50 \
    --prediction_type="v_prediction" \
    --snr_gamma=5 \
    --noise_offset=0 \
    --input_perturbation=0 \
    --conditioning="input" \
    --bias_he_ihc=0.5 \
    --output_dir=$OUTPUT_DIR \
    --project="$PROJECT_NAME" \
    --name="$RUN_TITLE" \
    --description="$RUN_DESCRIPTION" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_batch_size=16 \
    --learning_rate=1.5e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=1000 \
    --mixed_precision="bf16" \
    --resolution=512 \
    --translation_prompt="IHC" \
    --he_generation_prompt="H&E" \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=0 \
    --report_to="wandb" \
    --checkpointing_steps=4000 \
    > $OUTPUT_DIR.log 2>&1 &

    # --report_to="wandb" \
    # --snr_gamma=3 \
    # --prediction_type    "epsilon", "v_prediction"
    # --conditioning       "input", "xattention", "combined"],