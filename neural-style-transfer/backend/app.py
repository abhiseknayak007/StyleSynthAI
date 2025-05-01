import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Load the model on startup to avoid loading it each time a request is made
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(img_path, max_dim=512):
    """Load and preprocess an image for style transfer"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    # Check if content and style images are provided
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Content and style images are required'}), 400
    
    content_file = request.files['content']
    style_file = request.files['style']
    
    # Save uploaded files
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
    
    content_file.save(content_path)
    style_file.save(style_path)
    
    try:
        # Load images
        content_image = load_img(content_path)
        style_image = load_img(style_path)
        
        # Perform style transfer
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
        
        # Convert tensor to image
        output_image = tensor_to_image(stylized_image)
        
        # Save the result
        result_filename = f"stylized_{os.path.basename(content_path)}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        output_image.save(result_path)
        
        # Return the image as response
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name=result_filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded files (optional)
        if os.path.exists(content_path):
            os.remove(content_path)
        if os.path.exists(style_path):
            os.remove(style_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Style transfer API is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)