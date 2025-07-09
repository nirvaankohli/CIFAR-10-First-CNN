import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # plain ResNet-18 adapted for CIFAR-10
        self.backbone = models.resnet18(pretrained=False, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


@st.cache_resource
def load_model(model_version: int = 3, num_classes: int = 10):
    """
    Caches the ResNet18CIFAR instance.
    If you bump `model_version`, Streamlit will reload from disk.
    """
    model = ResNet18CIFAR(num_classes)
    path = f"modelV{model_version}.pth"
    ckpt = torch.load(path, map_location="cpu")
    # unwrap if you saved {'state_dict': ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.backbone.load_state_dict(ckpt)
    model.eval()
    return model


st.title("🤖 CIFAR-10 Classification with ResNet-18 (≈92% acc)")
st.write(
    """
    **CIFAR-10** is a dataset of 60,000 32×32 colour images in 10 classes.  
    This demo uses a **ResNet-18** backbone, achieving around **92%** test accuracy.

    **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """
)

uploaded = st.file_uploader(
    "Upload a CIFAR-10 image (one of the 10 classes above)", 
    type=["jpg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    # choose your version here; bump to bust cache
    MODEL_VERSION = 3

    with st.spinner(f"Loading model v{MODEL_VERSION}…"):
        model = load_model(model_version=MODEL_VERSION, num_classes=10)
    st.success("Model loaded!")

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225])
    ])
    inp = preprocess(img).unsqueeze(0)

    with st.spinner("Classifying…"):
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze()
            top5 = torch.topk(probs, 5)
    st.success("Done!")

    labels = top5.indices.tolist()
    confidences = top5.values.tolist()

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # 3-column metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Top-1",      class_names[labels[0]])
    c2.metric("Confidence", f"{confidences[0]*100:.1f}%")
    c3.metric("2nd-Best",   class_names[labels[1]])

    # top-5 table
    df = pd.DataFrame({
        "Rank":       list(range(1, 6)),
        "Class":      [class_names[i] for i in labels],
        "Confidence": [f"{p*100:.1f}%" for p in confidences]
    })
    st.table(df)

    # feedback
    st.subheader("Was our prediction correct?")
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None

    yes, no = st.columns(2)
    if yes.button("Yes", use_container_width=True):
        st.session_state.feedback = "yes"
    if no.button("No", use_container_width=True):
        st.session_state.feedback = "no"

    if st.button("Submit", use_container_width=True):
        if st.session_state.feedback:
            st.success("✔️ Thank you for your feedback!")
        else:
            st.warning("Please select Yes or No before submitting.")

# Hackathon badge
components.html(
    """
    <div align="center">
        <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
            <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
                alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
                style="width: 75%;">
        </a>
    </div>
    """,
    height=200
)
