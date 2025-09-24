import gradio as gr
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

# SAM 모델 로드
sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"  # 체크포인트 경로
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

def segment_object(image, x_coord, y_coord):
    # 이미지 전처리
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # 클릭 좌표 처리
    try:
        x, y = int(x_coord), int(y_coord)
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1: 객체 내부
    except ValueError:
        return "Error: 유효한 숫자 좌표를 입력하세요.", None

    # 세그멘테이션 예측
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    # 최상위 마스크 선택
    mask = masks[0].astype(np.uint8) * 255
    
    # 마스크 부드럽게 처리 (옵션)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 배경 제거
    object_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # 결과 이미지 저장
    output_path = "segmented_object.png"
    cv2.imwrite(output_path, cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR))
    
    return "객체가 성공적으로 세그멘테이션되었습니다!", output_path

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=segment_object,
    inputs=[
        gr.Image(type="pil", label="이미지 업로드"),
        gr.Number(label="X 좌표 (클릭 위치)"),
        gr.Number(label="Y 좌표 (클릭 위치)")
    ],
    outputs=[
        gr.Text(label="결과 메시지"),
        gr.Image(label="세그멘테이션된 객체")
    ],
    title="Segment Anything Model - 객체 세그멘테이션 MVP",
    description="이미지를 업로드하고 클릭 좌표를 입력하면 해당 객체를 배경에서 분리합니다."
)

# 인터페이스 실행
if __name__ == "__main__":
    iface.launch()