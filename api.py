import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import timm
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# ------------------ FastAPI App Setup ------------------
app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="EfficientNet-B0 with Grad-CAM Visualization",
    version="1.0"
)

# ------------------ Device & Model Initialization ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the same model architecture
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
# weights_only=True added to silence the FutureWarning
model.load_state_dict(torch.load('best_brain_tumor_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# Class names 
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Test Transform ------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- target layer for timm EfficientNet-B0 ---
# model.conv_head is the most stable target for Grad-CAM across different timm versions
target_layer = model.conv_head 

# ------------------ Grad-CAM Logic ------------------
def generate_gradcam(image: Image.Image):
    # Preprocess
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # Register hooks inside function to prevent state issues
    gradients = []
    activations = []

    def save_activation(module, input, output):
        activations.append(output.detach())

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(save_activation)
    h2 = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.eval()
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

        # Backward pass on predicted class
        model.zero_grad()
        output[0, pred_idx].backward()

        # Get gradients and activations
        grads = gradients[0].cpu().data.numpy().squeeze()
        acts = activations[0].cpu().data.numpy().squeeze()

        # Compute weights and CAM
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU, normalize and resize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Apply colormap and convert to RGB for overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Original image as numpy (RGB)
        original = np.array(image.resize((224, 224)))

        # Overlay Heatmap on Original (60% Original, 40% Heatmap)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        # Encode to PNG bytes (OpenCV expects BGR for encoding)
        success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")

        return buffer.tobytes(), class_names[pred_idx], round(confidence * 100, 2)

    finally:
        # Crucial: Always remove hooks to prevent memory leaks
        h1.remove()
        h2.remove()

# API Endpoints

@app.get("/")
def home():
    return {"message": "Brain Tumor Classifier API is running!"}

@app.post("/predict/visualize")
async def predict_and_visualize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generate Grad-CAM image and metadata
        gradcam_bytes, predicted_class, confidence = generate_gradcam(image)

        # Return the image with prediction data in custom headers
        return StreamingResponse(
            io.BytesIO(gradcam_bytes),
            media_type="image/png",
            headers={
                "X-Predicted-Class": predicted_class,
                "X-Confidence": f"{confidence:.2f}",
                "Access-Control-Expose-Headers": "X-Predicted-Class, X-Confidence"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# THE FASTAPI DOC view comes in handy for quick testing and prototyping api's
