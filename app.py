# app.py — Dogs vs Cats 분류 웹 서비스
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# — 모델 설정 ——————————————————————
# Google Drive 파일 ID: 1hYJCY2A-FkjjSko8k7dGypEhDf0tx4Sm
MODEL_URL = "https://drive.google.com/uc?id=1hYJCY2A-FkjjSko8k7dGypEhDf0tx4Sm"
MODEL_PATH = "best_model.pt"

@st.cache_resource
def load_model():
    # 1. 모델 파일 다운로드 (파일이 없거나 크기가 0인 경우)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        try:
            # 기존에 잘못된 파일이 있을 수 있으므로 삭제 후 재다운로드
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"모델 다운로드 중 오류 발생: {e}")
            return None

    # 2. ResNet18 모델 구조 정의
    # 분석 결과: 업로드된 모델은 표준 ResNet18 구조를 따릅니다.
    model = models.resnet18(weights=None)
    
    # 3. 출력 레이어 수정 (이진 분류: 1개 출력)
    # 분석 결과: fc.weight의 shape이 [1, 512]이므로 out_features=1입니다.
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    try:
        # 4. 가중치 로드
        # map_location='cpu'를 사용하여 GPU가 없는 환경에서도 실행 가능하게 합니다.
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.info("해결 방법: 로컬의 'best_model.pt' 파일을 삭제하고 앱을 새로고침하여 다시 다운로드 받으세요.")
        return None

# 모델 로드 실행
model = load_model()

# — 이미지 전처리 (ResNet 표준 전처리) ——————————————————————
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# — 페이지 설정 ——————————————————————
st.set_page_config(page_title="Dogs vs Cats 분류기", page_icon="🐶")
st.title("🐶 Dogs vs Cats 분류기")
st.caption("이미지를 업로드하면 개인지 고양이인지 분류합니다.")

# — 이미지 업로드 ——————————————————————
uploaded = st.file_uploader(
    "이미지를 선택하세요", type=["jpg", "jpeg", "png"]
)

if uploaded is not None and model is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    # — 예측 ——————————————————————
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logit = model(input_tensor)
        # Sigmoid를 통해 0~1 사이의 확률값 계산
        prob = torch.sigmoid(logit).item()

    # — 결과 표시 ——————————————————————
    # 0.5 이상이면 Dog(개), 미만이면 Cat(고양이)
    is_dog = prob >= 0.5
    label = "🐶 Dog" if is_dog else "🐱 Cat"
    confidence = prob if is_dog else 1 - prob

    st.markdown(f"### 예측 결과: {label}")
    st.metric("확신도", f"{confidence:.1%}")

    # 확률 바 시각화
    st.progress(prob if is_dog else 1 - prob)
elif model is None:
    st.warning("모델이 로드되지 않았습니다. 설정을 확인해주세요.")
    # 확률 바 시각화
    st.progress(prob if is_dog else 1 - prob)
elif model is None:
    st.warning("모델이 로드되지 않았습니다. 설정을 확인해주세요.")
