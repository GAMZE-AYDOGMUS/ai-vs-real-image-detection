import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/best_model.pt"
CLASS_NAMES = ["FAKE", "REAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def create_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = create_model()

def predict(image, uncertainty_threshold=0.75):
    if image is None:
        return {}, "Please upload an image."

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    scores = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    predicted_class = max(scores, key=scores.get)
    confidence = scores[predicted_class]

    if confidence < uncertainty_threshold:
        detail = f"Prediction: UNCERTAIN | Confidence: {confidence:.2%}"
    else:
        detail = f"Prediction: {predicted_class} | Confidence: {confidence:.2%}"

    return scores, detail

with gr.Blocks(title="AI vs Real Image Detection") as demo:
    gr.Markdown("# Yapay mı Gerçek mi Görsel Tespit Sistemi")
    gr.Markdown("Bir görsel yükleyin. Model görselin FAKE mi REAL mi olduğunu tahmin eder.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Görsel Yükle")
            threshold = gr.Slider(0.50, 0.95, value=0.75, step=0.01, label="Kararsızlık Eşiği")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            label_output = gr.Label(label="Sınıf Olasılıkları")
            text_output = gr.Textbox(label="Tahmin Detayı")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, threshold],
        outputs=[label_output, text_output]
    )

demo.launch()