import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from class_indices import class_indices 
from disease_info import disease_info

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), ## kernel size = 2, stride = 2

            nn.Conv2d(32,64, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),       
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(28*28*128, 256),
            nn.ReLU(),

            nn.Linear(256,37)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # # flattening
        x = self.fc_layers(x)

        return x
    
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()
   

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(image):
    img = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        probs = torch.softmax(outputs, dim=1)
        confidence = probs[0][predicted.item()].item() * 100

        class_name = class_indices[predicted.item()]
        
        # Split into crop and disease
        if "__" in class_name:
            crop, disease = class_name.split("__", 1)
        else:
            crop, disease = class_name, "Unknown"
        
        return crop, disease, confidence
    
st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱", layout="centered")

import streamlit as st
from PIL import Image


# ------------------- CSS -------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: #0e1117 !important;
}

/* Remove header */
header {visibility: hidden;}

/* Reduce top padding */
.block-container {
    padding-top: 1.5rem;
}

/* Title */
.main-title {
    text-align: center;
    color: white !important;
    font-size: 42px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #d3f9d8 !important;
    margin-bottom: 20px;
}

/* Card design */
.card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Image caption */
.stImage > div > div > div {
    font-size: 18px;
    font-weight: bold;
    color: white !important;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ------------------- TITLE -------------------
st.markdown("<h1 class='main-title'>🌱 AI-Powered Plant Health Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a leaf image and get instant disease analysis</p>", unsafe_allow_html=True)


# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((250, 250))

    # prediction first
    crop, disease, confidence = predict(image)

    # create columns
    col1, col2 = st.columns([1, 1])

    # LEFT SIDE → image
    with col1:
        st.markdown("#### ☘️ Leaf Under Analysis")
        st.image(image, caption="✨ Leaf Uploaded Successfully")

    # RIGHT SIDE → result
    with col2:
        st.markdown("#### 🧠 AI Diagnosis Result")
        st.markdown(f"""
        <div class="card">
            <h2 style="margin-bottom:5px;">🌿 {crop}</h2>
            <h4 style="color:#9be7a1 !important;">🦠 {disease}</h4>
            <p>📊 Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # AI verdict

        st.markdown(
            """
            <style>
            /* Success box styling */
            .stSuccess {
                margin-top: 20px;   /* উপরে gap */
                font-size: 18px;    /* font বড় */
                opacity: 0.80;      /* transparency */
            }

            /* Error box styling */
            .stError {
                margin-top: 30px;   /* উপরে gap */
                font-size: 18px;    /* font বড় */
                opacity: 0.80;      /* transparency */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if "healthy" in disease.lower():
            st.success("✅ AI Verdict: Plant is Healthy")
        else:
            st.error("⚠️ AI Verdict: Disease Detected")


# ------------------- INFO SECTION -------------------
    class_key = f"{crop}__{disease}"

    if class_key in disease_info:

        st.markdown("### 🧬 Diagnosis Details")

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🧬 Causes", "💡 Solution"])

        with tab1:
            st.markdown(f"""
            <div class="card">
                <h4>🧬 Causes</h4>
                <p>{disease_info[class_key]["causes"]}</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
            <div class="card">
                <h4>💡 How to Solve</h4>
                <p>{disease_info[class_key]["solution"]}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No extra information available yet for this disease.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        f"🧠 *AI Insight:* The model strongly predicts this condition based on leaf texture and color patterns."
    )


# ------------------- FOOTER -------------------
st.markdown(
    "<hr><center style='color:white !important;'>🌱 Powered by CNN | Developed by Masudur Rahman</center>",
    unsafe_allow_html=True
)

