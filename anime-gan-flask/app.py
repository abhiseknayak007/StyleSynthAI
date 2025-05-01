import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import torchvision.utils as vutils
from PIL import Image
import base64
import io
import uuid

app = Flask(__name__)

# Create directories if they don't exist
os.makedirs('static/images', exist_ok=True)

# Define Generator class (same as in your notebook)
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load the trained generator model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    
    # Try to load the model
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If model doesn't exist, we'll just use the untrained model
        print("Using untrained model")
    
    return generator, device

# Generate images using the model
def generate_image(generator, device, num_images=1, save=True):
    # Generate random noise
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = generator(noise)
    
    # Convert to numpy array
    images = fake_images.cpu().detach()
    
    # Save images and get paths
    image_paths = []
    
    if save:
        for i in range(num_images):
            img = images[i]
            # Normalize image
            img = (img + 1) / 2.0  # Convert from [-1,1] to [0,1]
            img = img.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
            img = (img * 255).astype(np.uint8)
            
            # Convert to PIL Image and save
            pil_img = Image.fromarray(img)
            filename = f"anime_{uuid.uuid4()}.png"
            filepath = os.path.join("static", "images", filename)
            pil_img.save(filepath)
            image_paths.append(filepath)
    
    # For API response, also return base64 representation
    grid = vutils.make_grid(images, normalize=True)
    grid = grid.permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    
    # Convert to base64 for API response
    pil_grid = Image.fromarray(grid)
    buffered = io.BytesIO()
    pil_grid.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return image_paths, img_str

# Define model path - you'll need to update this to where you save your trained model
MODEL_PATH = "models/generator.pth"

# Load the model at startup
generator, device = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get number of images to generate (default to 1)
        num_images = int(request.form.get('num_images', 1))
        num_images = min(max(1, num_images), 9)  # Limit to between 1 and 9 images
        
        # Generate images
        image_paths, img_base64 = generate_image(generator, device, num_images)
        
        # Return image paths and base64 data
        return jsonify({
            'success': True,
            'message': f'Generated {num_images} anime faces',
            'image_paths': ['/'+path for path in image_paths],
            'image_base64': img_base64
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating images: {str(e)}'
        })

@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/train', methods=['POST'])
def train_model():
    """
    Placeholder for training functionality
    In a real app, you'd want to set up a background task for this
    """
    return jsonify({
        'success': False,
        'message': 'Training functionality not implemented in this demo'
    })

if __name__ == '__main__':
    app.run(debug=True)