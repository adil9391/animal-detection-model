import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms

# Load your trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def load_image():
    filepath = filedialog.askopenfilename()
    img = Image.open(filepath)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    img_label.filepath = filepath


def detect_animals():
    # Load and preprocess image
    img = Image.open(img_label.filepath)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    # Run model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Process predictions (e.g., draw bounding boxes)
    # (This part would involve additional image processing)


root = tk.Tk()
root.title("Animal Detection")

img_label = tk.Label(root)
img_label.pack()

load_btn = tk.Button(root, text="Load Image", command=load_image)
load_btn.pack()

detect_btn = tk.Button(root, text="Detect Animals", command=detect_animals)
detect_btn.pack()

root.mainloop()
