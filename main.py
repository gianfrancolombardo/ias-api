from fastapi import FastAPI

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

app = FastAPI()

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/generate")
async def generate_image(prompt: str):
    image = pipe(prompt).images[0]
    image_bytes = image.getvalue()
    return {"image": image_bytes}