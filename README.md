# onetrainer cheat sheet


# Install
- https://github.com/Nerogar/OneTrainer
```shell
cd ~/Projects

git clone git clone https://github.com/Nerogar/OneTrainer.git
cd OneTrainer

# pyenv install 3.10
rm -rf venv 
pyenv local 3.10
./install.sh
./start-ui.sh
```

<br><br>

# Re-start
```shell
cd ~/Projects/OneTrainer

deactivate
rm -rf venv

pyenv local 3.10

./start-ui.sh
```




<br><br>
<br><br>

# Guides
- https://github.com/Nerogar/OneTrainer/blob/master/docs/QuickStartGuide.md
- https://github.com/Nerogar/OneTrainer/wiki

- https://www.youtube.com/watch?v=0t5l6CP9eBg
  - https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Tutorials/OneTrainer-Master-SD-1_5-SDXL-Windows-Cloud-Tutorial.md





<br><br>
<br><br>


#### Good 2 Know
- OneTrainer does not support DreamBooth training






<br><br>
<br><br>


#### WIKI

<br><br>

##### Workspaces
- OneTrainer uses workspaces to separate your training runs. A workspace contains all the backup, sampling and tensorboard data of a single run. When you start a new training run, it is recommended that you select an empty directory as your workspace.
- It is recommended to use new cache folder when you try different settings

<br><br>

#### Photo realistic checkpoints
SDXL - RealVisXL_V4 : https://huggingface.co/SG161222/RealVisXL_V4.0/tree/main
SD 1.5 - Hyper Realism V3 : https://civitai.com/api/download/models/292213?token=c4ac691fbdba136b3da555eed088fcbe














<br><br>
<br><br>

# Step by Step Guide

<br><br>

## Photorealistic Person

### SDXL 1.0
```javascript
{
    "__version": 6,
    "training_method": "FINE_TUNE",
    "model_type": "STABLE_DIFFUSION_XL_10_BASE",
    "debug_mode": false,
    "debug_dir": "debug",
    "workspace_dir": "/home/t33n/Projects/ai/workspace_train",
    "cache_dir": "/home/t33n/Projects/ai/workspace_train/cache",
    "tensorboard": false,
    "tensorboard_expose": false,
    "validation": false,
    "validate_after": 1,
    "validate_after_unit": "EPOCH",
    "continue_last_backup": false,
    "include_train_config": "NONE",
    "base_model_name": "/home/t33n/Projects/ai/resources/checkpoints/sd/sdxl/realistic/RealVisXL_V4.0.safetensors",
    "weight_dtype": "BFLOAT_16",
    "output_dtype": "BFLOAT_16",
    "output_model_format": "SAFETENSORS",
    "output_model_destination": "/home/t33n/Projects/ai/resources/checkpoints/sd/sdxl/self_trained/realistic/model.safetensors",
    "gradient_checkpointing": "ON",
    "force_circular_padding": false,
    "concept_file_name": "training_concepts/concepts.json",
    "concepts": [
        {
            "__version": 1,
            "image": {
                "__version": 0,
                "enable_crop_jitter": false,
                "enable_random_flip": false,
                "enable_fixed_flip": false,
                "enable_random_rotate": false,
                "enable_fixed_rotate": false,
                "random_rotate_max_angle": 0.0,
                "enable_random_brightness": false,
                "enable_fixed_brightness": false,
                "random_brightness_max_strength": 0.0,
                "enable_random_contrast": false,
                "enable_fixed_contrast": false,
                "random_contrast_max_strength": 0.0,
                "enable_random_saturation": false,
                "enable_fixed_saturation": false,
                "random_saturation_max_strength": 0.0,
                "enable_random_hue": false,
                "enable_fixed_hue": false,
                "random_hue_max_strength": 0.0,
                "enable_resolution_override": false,
                "resolution_override": "512",
                "enable_random_circular_mask_shrink": false,
                "enable_random_mask_rotate_crop": false
            },
            "text": {
                "__version": 0,
                "prompt_source": "concept",
                "prompt_path": "/home/t33n/Documents/pics me ai/caption.txt",
                "enable_tag_shuffling": false,
                "tag_delimiter": ",",
                "keep_tags_count": 1
            },
            "name": "training dennis",
            "path": "/home/t33n/Documents/pics me ai",
            "seed": -259049126,
            "enabled": true,
            "validation_concept": false,
            "include_subdirectories": false,
            "image_variations": 1,
            "text_variations": 1,
            "balancing": 1.0,
            "balancing_strategy": "REPEATS",
            "loss_weight": 1.0
        },
        {
            "__version": 1,
            "image": {
                "__version": 0,
                "enable_crop_jitter": false,
                "enable_random_flip": false,
                "enable_fixed_flip": false,
                "enable_random_rotate": false,
                "enable_fixed_rotate": false,
                "random_rotate_max_angle": 0.0,
                "enable_random_brightness": false,
                "enable_fixed_brightness": false,
                "random_brightness_max_strength": 0.0,
                "enable_random_contrast": false,
                "enable_fixed_contrast": false,
                "random_contrast_max_strength": 0.0,
                "enable_random_saturation": false,
                "enable_fixed_saturation": false,
                "random_saturation_max_strength": 0.0,
                "enable_random_hue": false,
                "enable_fixed_hue": false,
                "random_hue_max_strength": 0.0,
                "enable_resolution_override": false,
                "resolution_override": "512",
                "enable_random_circular_mask_shrink": false,
                "enable_random_mask_rotate_crop": false
            },
            "text": {
                "__version": 0,
                "prompt_source": "concept",
                "prompt_path": "/home/t33n/Projects/ai/resources/stable-diffusion-regularization-images/sdxl/man/caption.txt",
                "enable_tag_shuffling": false,
                "tag_delimiter": ",",
                "keep_tags_count": 1
            },
            "name": "regularization",
            "path": "/home/t33n/Projects/ai/resources/stable-diffusion-regularization-images/sdxl/man",
            "seed": 533657394,
            "enabled": true,
            "validation_concept": false,
            "include_subdirectories": false,
            "image_variations": 1,
            "text_variations": 1,
            "balancing": 0.0022,
            "balancing_strategy": "REPEATS",
            "loss_weight": 1.0
        }
    ],
    "aspect_ratio_bucketing": false,
    "latent_caching": true,
    "clear_cache_before_training": true,
    "learning_rate_scheduler": "CONSTANT",
    "custom_learning_rate_scheduler": null,
    "scheduler_params": [],
    "learning_rate": 1e-05,
    "learning_rate_warmup_steps": 200,
    "learning_rate_cycles": 1,
    "epochs": 400,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "ema": "OFF",
    "ema_decay": 0.999,
    "ema_update_step_interval": 1,
    "dataloader_threads": 2,
    "train_device": "cuda",
    "temp_device": "cpu",
    "train_dtype": "BFLOAT_16",
    "fallback_train_dtype": "FLOAT_32",
    "enable_autocast_cache": false,
    "only_cache": false,
    "resolution": "1024",
    "attention_mechanism": "DEFAULT",
    "align_prop": false,
    "align_prop_probability": 0.1,
    "align_prop_loss": "AESTHETIC",
    "align_prop_weight": 0.01,
    "align_prop_steps": 20,
    "align_prop_truncate_steps": 0.5,
    "align_prop_cfg_scale": 7.0,
    "mse_strength": 1.0,
    "mae_strength": 0.0,
    "log_cosh_strength": 0.0,
    "vb_loss_strength": 1.0,
    "loss_weight_fn": "CONSTANT",
    "loss_weight_strength": 5.0,
    "dropout_probability": 0.0,
    "loss_scaler": "NONE",
    "learning_rate_scaler": "NONE",
    "offset_noise_weight": 0.0,
    "perturbation_noise_weight": 0.0,
    "rescale_noise_scheduler_to_zero_terminal_snr": false,
    "force_v_prediction": false,
    "force_epsilon_prediction": false,
    "min_noising_strength": 0.0,
    "max_noising_strength": 1.0,
    "timestep_distribution": "UNIFORM",
    "noising_weight": 0.0,
    "noising_bias": 0.5,
    "unet": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": 10000,
        "stop_training_after_unit": "NEVER",
        "learning_rate": 1e-05,
        "weight_dtype": "BFLOAT_16",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "prior": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": 0,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "text_encoder": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": 10000,
        "stop_training_after_unit": "NEVER",
        "learning_rate": 3e-06,
        "weight_dtype": "BFLOAT_16",
        "dropout_probability": 0.0,
        "train_embedding": false,
        "attention_mask": false
    },
    "text_encoder_layer_skip": 0,
    "text_encoder_2": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": false,
        "stop_training_after": 0,
        "stop_training_after_unit": "EPOCH",
        "learning_rate": null,
        "weight_dtype": "BFLOAT_16",
        "dropout_probability": 0.0,
        "train_embedding": false,
        "attention_mask": false
    },
    "text_encoder_2_layer_skip": 0,
    "text_encoder_3": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": 30,
        "stop_training_after_unit": "EPOCH",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "text_encoder_3_layer_skip": 0,
    "vae": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "FLOAT_32",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "effnet_encoder": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "decoder": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "decoder_text_encoder": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "decoder_vqgan": {
        "__version": 0,
        "model_name": "",
        "include": true,
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "learning_rate": null,
        "weight_dtype": "NONE",
        "dropout_probability": 0.0,
        "train_embedding": true,
        "attention_mask": false
    },
    "masked_training": true,
    "unmasked_probability": 0.1,
    "unmasked_weight": 0.6,
    "normalize_masked_area_loss": false,
    "embedding_learning_rate": null,
    "preserve_embedding_norm": false,
    "embedding": {
        "__version": 0,
        "uuid": "ca4a8f60-4077-4baa-b35f-50eceea29885",
        "model_name": "",
        "placeholder": "<embedding>",
        "train": true,
        "stop_training_after": null,
        "stop_training_after_unit": "NEVER",
        "token_count": 1,
        "initial_embedding_text": "*"
    },
    "additional_embeddings": [],
    "embedding_weight_dtype": "FLOAT_32",
    "peft_type": "LORA",
    "lora_model_name": "",
    "lora_rank": 16,
    "lora_alpha": 1.0,
    "lora_decompose": false,
    "lora_decompose_norm_epsilon": true,
    "lora_weight_dtype": "FLOAT_32",
    "lora_layers": "attentions",
    "lora_layer_preset": "attn-mlp",
    "bundle_additional_embeddings": true,
    "optimizer": {
        "__version": 0,
        "optimizer": "ADAFACTOR",
        "adam_w_mode": false,
        "alpha": null,
        "amsgrad": false,
        "beta1": null,
        "beta2": null,
        "beta3": null,
        "bias_correction": false,
        "block_wise": false,
        "capturable": false,
        "centered": false,
        "clip_threshold": 1.0,
        "d0": null,
        "d_coef": null,
        "dampening": null,
        "decay_rate": -0.8,
        "decouple": false,
        "differentiable": false,
        "eps": 1e-30,
        "eps2": 0.001,
        "foreach": false,
        "fsdp_in_use": false,
        "fused": false,
        "fused_back_pass": false,
        "growth_rate": null,
        "initial_accumulator_value": null,
        "is_paged": false,
        "log_every": null,
        "lr_decay": null,
        "max_unorm": null,
        "maximize": false,
        "min_8bit_size": null,
        "momentum": null,
        "nesterov": false,
        "no_prox": false,
        "optim_bits": null,
        "percentile_clipping": null,
        "r": null,
        "relative_step": false,
        "safeguard_warmup": false,
        "scale_parameter": false,
        "stochastic_rounding": false,
        "use_bias_correction": false,
        "use_triton": false,
        "warmup_init": false,
        "weight_decay": 0.01,
        "weight_lr_power": null,
        "decoupled_decay": false,
        "fixed_decay": false,
        "rectify": false,
        "degenerated_to_sgd": false,
        "k": null,
        "xi": null,
        "n_sma_threshold": null,
        "ams_bound": false,
        "adanorm": false,
        "adam_debias": false
    },
    "optimizer_defaults": {
        "ADAFACTOR": {
            "__version": 0,
            "optimizer": "ADAFACTOR",
            "adam_w_mode": false,
            "alpha": null,
            "amsgrad": false,
            "beta1": null,
            "beta2": null,
            "beta3": null,
            "bias_correction": false,
            "block_wise": false,
            "capturable": false,
            "centered": false,
            "clip_threshold": 1.0,
            "d0": null,
            "d_coef": null,
            "dampening": null,
            "decay_rate": -0.8,
            "decouple": false,
            "differentiable": false,
            "eps": 1e-30,
            "eps2": 0.001,
            "foreach": false,
            "fsdp_in_use": false,
            "fused": false,
            "fused_back_pass": false,
            "growth_rate": null,
            "initial_accumulator_value": null,
            "is_paged": false,
            "log_every": null,
            "lr_decay": null,
            "max_unorm": null,
            "maximize": false,
            "min_8bit_size": null,
            "momentum": null,
            "nesterov": false,
            "no_prox": false,
            "optim_bits": null,
            "percentile_clipping": null,
            "r": null,
            "relative_step": false,
            "safeguard_warmup": false,
            "scale_parameter": false,
            "stochastic_rounding": false,
            "use_bias_correction": false,
            "use_triton": false,
            "warmup_init": false,
            "weight_decay": 0.01,
            "weight_lr_power": null,
            "decoupled_decay": false,
            "fixed_decay": false,
            "rectify": false,
            "degenerated_to_sgd": false,
            "k": null,
            "xi": null,
            "n_sma_threshold": null,
            "ams_bound": false,
            "adanorm": false,
            "adam_debias": false
        }
    },
    "sample_definition_file_name": "training_samples/samples.json",
    "samples": [
        {
            "__version": 0,
            "enabled": true,
            "prompt": "hyper-realistic photo of a smiling 27-year-old man with short brown hair lying on the floor of a luxurious penthouse apartment, surrounded by adorable kittens playing on him, capturing their playful interaction, with the man looking directly into the camera, sharp facial details, deep texture, ultra-high-definition, natural lighting",
            "negative_prompt": "blurry, grainy, low-res, cartoonish, overexposed, unrealistic, digital artifacts, unnatural lighting, over-sharpened, harsh shadows",
            "height": 1024,
            "width": 1024,
            "seed": 42,
            "random_seed": true,
            "diffusion_steps": 40,
            "cfg_scale": 7.0,
            "noise_scheduler": "DPMPP_SDE_KARRAS",
            "text_encoder_1_layer_skip": 0,
            "text_encoder_2_layer_skip": 0,
            "text_encoder_3_layer_skip": 0,
            "force_last_timestep": false,
            "sample_inpainting": false,
            "base_image_path": "",
            "mask_image_path": ""
        },
        {
            "__version": 0,
            "enabled": true,
            "prompt": "hyper-realistic photo of a smiling 27-year-old man with short brown hair in black suite, with the man looking directly into the camera, sharp facial details, deep texture, ultra-high-definition, natural lighting",
            "negative_prompt": "blurry, grainy, low-res, cartoonish, overexposed, unrealistic, digital artifacts, unnatural lighting, over-sharpened, harsh shadows",
            "height": 1024,
            "width": 1024,
            "seed": 42,
            "random_seed": true,
            "diffusion_steps": 40,
            "cfg_scale": 7.0,
            "noise_scheduler": "DPMPP_SDE_KARRAS",
            "text_encoder_1_layer_skip": 0,
            "text_encoder_2_layer_skip": 0,
            "text_encoder_3_layer_skip": 0,
            "force_last_timestep": false,
            "sample_inpainting": false,
            "base_image_path": "",
            "mask_image_path": ""
        }
    ],
    "sample_after": 10,
    "sample_after_unit": "MINUTE",
    "sample_image_format": "JPG",
    "samples_to_tensorboard": false,
    "non_ema_sampling": false,
    "backup_after": 30,
    "backup_after_unit": "NEVER",
    "rolling_backup": false,
    "rolling_backup_count": 3,
    "backup_before_save": false,
    "save_every": 15,
    "save_every_unit": "EPOCH",
    "save_skip_first": 0,
    "save_filename_prefix": "dennis"
}
```

1. [GENERAL Tab]
- 1.1 Set new workspace directory and cache directory
- 1.2 Disable tensorboard not realy needed (TensorBoard generates real-time graphs that display the training metrics)



<br><br>

2. [MODEL TAB]
- 2.1 Select Base Model
  - SDXL- RealVisXL_V4 or SD 1.5 - Hyper Realism V3
  - **When you use custom models like this do not choose VAE**

- 2.11 **Choose `Stable Diffusion XL 1.0 Base` and `Fine Tune` at the top right**

- 2.2 Select final checkpoint path (Model Output Desitination) with output format Safetensors e.g.
  > /home/userName/Projects/ai/resources/checkpoints/sd/sdxl/self_trained/realistic/model.safetensors

- 2.3 Choose following options
     ```
     Weight Data Type: bfloat16
     Override Unet Data Type: bfloat16
     Override Text Encoder 1 Data Type: bfloat16
     Override Text Encoder 2 Data Type: bfloat16
     Override VAE Data Type: float32
     Output Data Type: bfloat16
     Include Config: None
     ```



<br><br>

3. [DATA TAB]
- 3.1 Enable `Latent Caching` 
- 3.2 If you have images with different resolutions then enable `Aspect Ratio Bucketing`
    - But it is recommended to try first with single resolution
- 3.3 If you are not changing the dataset and only trying different paramaters you can disable `Clear cache before training`



<br><br>

4. [CONCEPTS TAB]
- 4.1 Click `add concept` and then click on icon
- 4.11 Set name and enable

- 4.12 Select path of training images
  - Your training images should be all in the same size as your regularization images. You can resize if needed with:
     ```shell
     mogrify -resize 1024x1024 -background white -gravity center -extent 1024x1024 *
     ```

  - What is a good dataset for training images?
    - https://github.com/CyberT33N/machine-learning-cheat-sheet/blob/main/README.md#dataset 

- 4.13 Prompt source
- **If you have low amount of pictures like 10-20 images OR/AND if you are training a person it is not recommended to create captions for each image because it will reduce the likeliness of the model.**

<br>

- 4.14 [GENERAL TAB]
     - 4.14.1 a) [WITH CAPTIONING]
          - https://huggingface.co/docs/transformers/model_doc/kosmos-2
          - https://huggingface.co/microsoft/kosmos-2-patch14-224
          - https://github.com/CyberT33N/ImageCaptioner/tree/master

          So .txt file would look like e.g.:
          ```
          ohwx man, A man with dark hait and glasses is standing in a room
          ```
               - In this case ohwx man is the unique identifier

     <br>

     4.14.1 b) [WITHOUT CAPTIONING]
     - Choose `From single text file` and create new txt file with:
     ```
     ohwx man
     ```
     - `ohwx` is a rare toke, which will lean my unique characterisitics into this token
          - `rare token` means that there weren't many images during the initial training on this token
     - `man` is the class I'm going to train my characteristics on

     <br>

     4.14.2 We are not using Variations
     ```
     Image Variations: 1
     Text Variations: 1
     Balancing: 1.0 <-- In older version the name was Repeats
     Loss Weight: 1.0
     Choose Repeats
     ```
     - OneTrainer does not support DreamBooth training so we using filly fine-tuning
          - However, with make the effect of DreamBooth with another concept

     <br>

     - 4.14 [IMAGE AUGMENTATION TAB]
     - Disable anything

     <br>

     - 4.15 [TEXT AUGMENTATION TAB]
     - Disable anything

     <br>

     - 4.2 As said, OneTrainer does not support DreamBooth training so we using filly fine-tuning. However, with make the effect of DreamBooth with another concept by adding `regularization images concept`. So click `add concept` and add any name

     <br>

     4.2.1 Choose regularization images path where the image Resolution is same like your dataset images.
     - **You can train without regularization images but then the quality will be worser**
     - https://github.com/tobecwb/stable-diffusion-regularization-images/tree/main [FREE]
     - https://www.patreon.com/posts/massive-4k-woman-87700469 [PAID]
     > Regularization images help prevent overfitting during model training by stabilizing the learning process. These images typically contain general, non-specific content and share a similar resolution with the training images, prompting the model to learn broader features instead of memorizing specifics. By incorporating these images, the model becomes more generalized and robust, improving its ability to perform well on unseen data.

     <br>

     4.2.2 Choose `From single file text` and create file with:
     ```
     man
     ```
     - Because `man` is the class token as we train ourselves on the man class, the model will forget what is man because it will see only our images aś a man
     > This will not work with anime or 3d. If you do training with stylization, you may not use this dataset or you can extract Lora from the trained DreamBooth model and use it on a stylized model

     <br>

     4.2.3 We do want Repeat 1.0 here because then it will run for each image. To get the correct number you can calculate it with `training images number / Regularization images number`. In my case `36/5000=0.0072`. If longer then x.xxxx then increase last number
          ```
          Image Variations: 1
          Text Variations: 1
          Balancing: 0.0072
          Loss Weight: 1.0
          Choose Repeats
          ```

     <br>

     4.2.4 [IMAGE AUGMENTATION TAB]
          - Disable anything







<br><br>

- 5. [TRAINING TAB]
     ```
     Optimizer: ADAFACTOR
     EPS: 1e-30
     Clip Thresold: 1.0
     EPS 2: 0.001
     Decay Rate : -0.8
     Weight Decay: 0.01

     Enable:
     - Fused Back Pass
          - **Allows to train with only 10GB VRAM RAM. If you have higher VRAM then do not enble it**

     Disable:
     - Scale Parameter
     - Warmup Initilizaton
     - Relative Step
     - Stochastic Round

     Learning Rate Scheduler: CONSTANT

     Learning Rate: 1e-05

     Learning Rate Warmup Steps: 200
     - Doesnt matter because we dont use a linear, cosine or anything, we are using CONSTANT

     Learning Rate Cycles: 1

     Epochs: 200
     - There is not a number that works for all, but 200 is usually good. You can try 400 and save frequent checkpoints and compare them
     - Let's say you are training yourself with 100 images, then 200 epochs may cause overtraining.
     - Or lets say you are training with 10 images, then 200 epochs may be not enough

     Batch Size: 1
     - If you need speed increase but 1 is the best quality for training

     Accumulation Steps: 1
     - Not working with Fused Back Pass from Optimizer

     Learning Rate Scaler: NONE

     ---------------------------------------

     Enable -> Train Text Encoder 1

     Disable -> Train Text Encoder 1 Embedding
     - Embeddings in a text encoder are dense vectors representing words or sentences to capture their meaning. They allow the model to understand semantic relationships, reduce dimensionality, and process text data efficiently. During training, embeddings are refined to improve the model's understanding of word relations, enabling it to perform tasks like classification or question answering more accurately.

     Stop Training After: 10000 never
     - You should use the same amount as you use EPOCHS for the model from above. 
     - If using 10000 never it will automatically use the EPOCHS which you set above

     Text Encoder 1 Learning Rate: 3e-06
     - Text encoder learning rate is lower than model learning rate. Be careful, if you use the same learning rate it will burn the model

     Clip Skip 1: 0
     - Usefully for anime or stylization

     ---------------------------------------

     Disable -> Train Text Encoder 2
     - It may cause overtraining to enable

     Stop Train After: 0 Epochs

     ---------------------------------------

     Attention: DEFAULT

     EMA: OFF
     - Very usefully for SD 1.5 but no benefit for SDXL

     EMA Decay: 0.999

     EMA Update Step Interval: 1

     Enable -> Gradien checkpointing
     - If you are training on A6000 GPU do not enable this. This wil speed up your training


     Train-Data Type: bfloat16

     Fallback rain-Data Type: float32

     Disable -> Autocast Cache

     Resolution: 1024
     - Make sure that you training images have the same size as your regularization images

     ---------------------------------------

     Enable -> Train UNet 

     Stop Training After: 10000 NEVER

     Unet Learning Rate: 1e-05
     - Same as model learning rate

     Disable -> Rescale Noise Scheduler

     ---------------------------------------

     Offset noise Weight: 0.0
     Perturbation Noise Weight: 0.0
     Min Noising Strength: 0.0
     Max Noising Strength: 1.0
     Noising Weight: 0.0
     Noising Blas: 0.5

     ---------------------------------------

     Disable -> AlignProp
     AlignProp Probability: 0.1
     AlignProp Loss: AESTHETIC
     AlignProp Weight: 0.01
     AlignProp Steps: 20
     AlignProp Truncate Steps: 0.5
     AlignProp CFG Scale: 7.0

     ---------------------------------------

     Enable -> Masked Training
     - If your dataset is not perfekt you will get better results 

     Unmasked Probability: 0.1

     Unmasked Weight: 0.6
     - Means non-masked areas will get 60% weight instead of 100%
     - So the background and my clothing will get 60% weight during training, and my head will get 100% weight

     Disable -> Normalize Masked Area Loss

     ---------------------------------------

     MSE Strength 1.0
     MAE Strength 0.0
     Loss Weight Function: CONSTANT
     Gamma: 5.0
     Loss Scaler: NONE
     ```


<br><br>

- 6. [TOOLS TAB]
 
  [MASKED TRAINING]
  - **Adding masked training will cause anatomy problems but the quality will be better when your dataset for training is low quality. So if you have a really good dataset do not use maskedm training and skip the steps below**

  - 6.1 Use `Dataset Tools` and open folder of training images

  - 6.2 Click `Generate Masks` and select prompt `head` and click `create mask`
    - It maybe take some time if you have high amount of pictures. There will be progress bar below. After finish you can click close


<br><br>

- 7. [SAMPLING TAB]
  - This is usefully when you want to see the point when the model gets overtrained

  - 7.1 You can generate samples during training. You can click on the right on the 3 dots to add more specific prompts like:
  ```
  prompt: hyper-realistic photo of an man, sharp facial details, deep texture, ultra-high-definition, natural lighting, professional studio background

  negative prompt: blurry, grainy, low-res, cartoonish, overexposed, unrealistic, digital artifacts, unnatural lighting, over-sharpened, harsh shadows

  steps 40
  cfg scale: 7-10
  enable: random seed
  sampler: DPM++ SDE
  Schedule Type: Karras
  ```

  7.2 Diasable `Non-EMA Sampling` and `Samples to tensorboard`




<br><br>

- 8. [BACKUP TAB]
     - An alternative to check your training state by using sample is to do x/y/z checkpoint comparsion at the end of the training
     ```
     Backup after: 30 NEVER

     Disable -> Rolling Backup
     Disable -> Backup before save

     Save after: 15 EPOCH
     - Means will generate 10 checkpoints. Each checkpoint is like 6.5GB
     - You can see it in `model tab` under `Output Data Type` because of bfloat 16 which means half precision

     Save filename prefix: tutorial_1
     ```

<br><br>

- 9. Save config

<br><br>

- 10. Check VRAM Usage while training
     ```shell
     sudo apt install pipx
     pipx install nvitop
     pipx ensurepath

     # Restart terminal then run
     nvitop
     ```
     - If possible kill all processes that you do not need which at the moment use vram

<br><br>

11. Start training
     - **Verify that you choosed `Stable Diffusion XL 1.0 Base` and ``Fine Tune` at the top right**
     - First it will cache the images then it will start training
     - In the terminal you and in onetrainer at the bottom left you can see how many epochs will be generated like e.g.
     ```
     epoch:   2%|▍                                | 3/200 [13:46<11:15:41, 205.79s/it]s=0.122]
     ```
     - You can calculate estimated time by:
     - amount of concepts (2)
     - amount of images (36) with repeating 1
     - amount of regularization images (5000) with reapeating 0.0072

          - 200 epoch x 72 images = 14,400 steps
          - If your speed is 1.13it/s then each step is 1 second
               - So 14,400 steps x 1 seonds / 60 = 240 minutes+



<br><br>

- 12. Start stable-diffusion-webui (http://127.0.0.1:7860)

  - 12.1 Load checkpoint `yourname.safetensors`

  - 12.1.2
    - Check your generated sample for best results and then choose a checkpoint
      - For some cases 150 epochs is a sweet spot. In my case 300 was good with 11 low images which ym cp was trained
  
      General:
      ```
      1024x1024
  
      prompt: (candid photography:1.2), 4k photograph of a ohwx men, casual business clothes,
      shot on Canon EOS R5, 85mm f/1.4 lens, unedited RAW photo,
      realistic skin texture,
      (crisp details:1.5), (high definition:2), (sharp focus)
      (soft shadows:1.25),
      (color temperature 4000K), (golden hour lighting:2), (natural lighting:5)
  
      negative: (hyper-detailed:2), (ultra high resolution), (8K), (HDR:2), (over-sharpened),
      (CGI), (plastic skin), (digital art),
      (smooth:3), (too perfect), (fake), (overexposed), (low contrast),
      (digital artifacts), (extra fingers), (mutated limbs),
      glasses
  
      sample: 40
      Sampling Method: DPM++ 2M SDE
      cfg scale: 7
      batch count 4
      ```
      - For alternative prompts check:
        - https://github.com/CyberT33N/stable-diffusion-cheat-sheet/blob/main/README.md#prompts  
    
      <br>
  
      AIDetailer:
      - Enable it to give better quality. It totally depends on your training how much it will make your result better
      ```
      prompt: photo of ohwx man slightly smiling
  
      detection set `Mask only the top k target:1`:
      - help you to detect face when there are multiple people
  
      inpainting:
      - Inpainting denosing strength 0.35
        - In my case everything above will crash above 0.5
        - Also notice the higher you go the more sharpen it will get and less natural it will look
  
      - ENABLEA AND Use seperate steps: 70
        - Will improve quality of the face
      ```
 
      <br>
  
      Upscaler (Hires, fix):
      - If needed you can upscale your image
        - **It maybe deform your face when using it. Be carefully**
      ```
      Choose:
        - R-ESRGAN 4x plus and upscale 1.25
        - Maybe change denoise strength
      ```

      <br>
  
      - SDXL Styles:
        - Dependening on your training data and model quality this style maybe make it looks good:
          - HDR
          - Hyperrealism
  
      - Extend if needed with using Loras..

  <br><br>
  
  - 13. Checkpoint comparsion
    - **For me with 11 not good Images the best result was checkpoint with epoch 300**
  
    - Go to bottom to scripts and choose `x/y/z plot`
      - X type: `checkpoint name` and select all checkpoints
      - Choose `50 grid margins` and at top choose `Batch count: 4` for 4 images
  
    - Images will be saved to stable-diffusion-webui/output/txt2img-images and to txt2img-grid
      - In txt2img-grid will be aswell the big png file
  
      - Check which checkpoint seems to be the best for you
        - If e.g. quality of clothing is degrading you know it is overtrained
          
  <br><br>
  
  - 14. Good 2 know
    - **In my case for 11 training images with 400 epochs the best checkpoint was at 300**
  
    - There is a trade-off between the likeliness and the flexibility
      - So if you want more likeliness you need to do more train and it will become less flexible
  
    - **If you want to add expressions like slight smiling then add it to the prompt in AIDetailer**
  
  <br><br>

  - 15. Inpaint
    - If you want to add e.g. manually face impressions and get better results then use inpaint by click on the color icon under your generated image and then you will be in the Inpaint area
      - Carefully mask the face and use prompt: `slightly smiling photo of ohwx man`
      - Select `only masked`
      - Select sampler DPM++ 2M SDE Karras, Steps 60, Denoising Strength 0.35
      - Make seed random that you get different results and pick the best one
      - 1024x1024



    
