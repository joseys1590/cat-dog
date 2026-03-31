import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(page_title="Dogs vs Cats 분류기", page_icon="🐾")

# ── 모델 로드 (캐싱) ─────────────────────────────────────
@st.cache_resource
def load_model():
    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    base.eval()
    return base

model = load_model()

# ImageNet 기준 고양이 / 강아지 클래스 인덱스
CAT_INDICES = list(range(281, 286))   # tabby, tiger cat, Persian cat …
DOG_INDICES = list(range(151, 269))   # 수백 개 품종

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]   # 1000개 클래스 확률

    cat_prob = probs[CAT_INDICES].sum().item()
    dog_prob = probs[DOG_INDICES].sum().item()
    total    = cat_prob + dog_prob + 1e-9

    cat_pct = cat_prob / total * 100
    dog_pct = dog_prob / total * 100
    label   = "cat" if cat_pct >= dog_pct else "dog"
    return label, cat_pct, dog_pct

# ── UI ───────────────────────────────────────────────────
st.title("🐾 Dogs vs Cats 분류기")
st.write("이미지를 업로드하면 강아지인지 고양이인지 분류합니다.")

uploaded = st.file_uploader("이미지를 선택하세요", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, use_container_width=True)

    with st.spinner("분석 중..."):
        label, cat_pct, dog_pct = predict(img)

    st.divider()

    if label == "cat":
        st.markdown("### 예측 결과: 🐱 Cat")
    else:
        st.markdown("### 예측 결과: 🐶 Dog")

    confidence = max(cat_pct, dog_pct)
    st.write(f"**확신도**")
    st.write(f"{confidence:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.write("🐱 Cat")
        st.progress(cat_pct / 100)
    with col2:
        st.write("🐶 Dog")
        st.progress(dog_pct / 100)
else:
    st.caption("왼쪽 위의 업로드 버튼을 눌러 이미지를 선택하세요.")
