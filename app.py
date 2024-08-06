import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token  # Assuming you have a file authtoken.py with auth_token defined
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
root = ctk.CTk()  # Using customtkinter as the main window
root.geometry("532x632")
root.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Input field for the prompt
prompt = ctk.CTkEntry(master=root, height=40, width=512, fg_color="white")
prompt.place(x=10, y=10)

# Label to display the generated image
lmain = ctk.CTkLabel(master=root, height=512, width=512)
lmain.place(x=10, y=110)

# Model configuration
model_id = "CompVis/stable-diffusion-v1-4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline with appropriate settings
if device.type == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=auth_token)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)

pipe.to(device)

def generate():
    with autocast(device.type):
        prompt_text = prompt.get()
        image = pipe(prompt_text, guidance_scale=8.5)["sample"][0]
        image.save('generatedimage.png')
        img = ImageTk.PhotoImage(image)
        lmain.imgtk = img  # Keep a reference to avoid garbage collection
        lmain.configure(image=img)

# Button to trigger image generation
trigger = ctk.CTkButton(master=root, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=60)

# Start the main loop
root.mainloop()
