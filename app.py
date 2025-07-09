import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import pandas as pd
from PIL import Image


class ResNet18CIFAR(nn.Module):

    def __init__(self, num_classes: int):

        super(ResNet18CIFAR, self).__init__()

        self.backbone = models.resnet18(pretrained=False, num_classes=num_classes)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            
        )

        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

@st.cache(allow_output_mutation=True)

def load_model(num_classes: int = 10):

    model = ResNet18CIFAR(num_classes)

    state = torch.load("model.pth", map_location="cpu")

    model.load_state_dict(state)

    model.eval()

    return model

st.title("🤖 CIFAR-10 Classification with ResNet-18 (≈80% acc)")

st.write(
    """
    **CIFAR-10** is a dataset of 60,000 32×32 colour images in 10 classes.  
    This demo uses a **ResNet-18** backbone, achieving around **80%** test accuracy.
   
    The following classes are included in the CIFAR-10 dataset:
        
        airplane										
        automobile										
        bird										
        cat										
        deer										
        dog										
        frog										
        horse										
        ship										
        truck	

    Demo is below :)									

      """
)

# ─── File uploader ────────────────────────────────────

uploaded = st.file_uploader("Upload a CIFAR-10 image(A image of one of the classes above.)", type=["jpg", "png"])

if uploaded:

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    # ─── Loading status ────────────────────────────────

    st.status("Loading model…")

    model = load_model(num_classes=10)

    st.status("Model loaded!")


    # ─── Preprocess + inference ───────────────────────

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225])
    ])

    st.status("Transforming & Processing Your Image")

    inp = preprocess(img).unsqueeze(0)

    st.status("Predicting the Class")

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).squeeze()
        top5 = torch.topk(probs, 5)
        labels = top5.indices.tolist()
        confidences = top5.values.tolist()

    st.status("loaded")

    # ─── 3-column metrics ──────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Top-1", f"Class {labels[0]}")
    col2.metric("Confidence", f"{confidences[0]*100:.1f}%")
    col3.metric("2nd-Best", f"Class {labels[1]}")

    # ─── Table of top-5 ────────────────────────────────
    df = pd.DataFrame({
        "Rank": list(range(1, 6)),
        "Class": labels,
        "Confidence": [f"{p*100:.1f}%" for p in confidences]
    })
    st.table(df)

    # ─── Feedback widget ───────────────────────────────
    st.subheader("Was our prediction correct?")
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None

    col_yes, col_no = st.columns(2)
    if col_yes.button("Yes"):
        st.session_state.feedback = "yes"
    if col_no.button("No"):
        st.session_state.feedback = "no"

    if st.button("Submit"):
        if st.session_state.feedback:
            st.success("✔️ Thank you for your feedback!")
        else:
            st.warning("Please select Yes or No before submitting.")

st.html(
"""
<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 35%;">
  </a>
</div>

"""
)