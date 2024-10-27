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

- 12. Start stable-diffusion-webui

  - 12.1 Load checkpoint `yourname.safetensors`

  - 12.1.1 Use:
  ```
  1024x1024
  Sampling Steps: 40
  CFG Scale: 7
  Sampling Method: DPM++ 2M SDE
  Schedule Type: Karras

  positive: hyper-realistic photo of a smiling 27-year-old man with short brown hair lying on the floor of a luxurious penthouse apartment, surrounded by adorable kittens playing on him, capturing their playful interaction, with the man looking directly into the camera, sharp facial details, deep texture, ultra-high-definition, natural lighting
  
  negative: blurry, grainy, low-res, cartoonish, overexposed, unrealistic, digital artifacts, unnatural lighting, over-sharpened, harsh shadows
  ```
