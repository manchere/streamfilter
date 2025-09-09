
import torch

from diffusers import AutoencoderTiny
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.pipeline import StreamDiffusion
from PIL import Image
import numpy as np
import cv2

# Initialize the pipeline and streamdiffusion model only once
# NOTE: If you still get errors about StreamDiffusion or AutoencoderTiny, check their documentation for correct usage.
pipeline = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

stream = StreamDiffusion(
    pipe=pipeline,
    t_index_list=[32, 45],
    torch_dtype=torch.float16,
)

stream.load_lcm_lora()
stream.fuse_lora()
stream.vae = AutoencoderTiny.from_pretrained("KBlueLeaf/tiny-vae").to(device=pipeline.device, dtype=pipeline.dtype)
pipeline.enable_xformers_memory_efficient_attention()

# You can set the prompt once, or expose a function to change it if needed
prompt = "girl with dog hair"
stream.prepare(prompt)

def stream_diffusion_frame(frame_bgr):
    """
    Takes a BGR frame (OpenCV), runs StreamDiffusion, and returns the processed BGR image.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    x_output = stream(pil_image)
    # If postprocess_image is a function, call it; if it's a class, use .process or similar
    result_imgs = postprocess_image(x_output, output_type="pil")
    # Handle both list and single image return
    if isinstance(result_imgs, list):
        result_img = result_imgs[0]
    else:
        result_img = result_imgs
    result_bgr = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
    return result_bgr