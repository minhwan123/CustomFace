"""GUI(app.py)에서 호출하는 캐리커처 생성 파이프라인.

이미지 경로와 사용자가 선택한 부위 목록을 받아 전처리 → 랜드마크 추출 →
부위별 왜곡 → 카툰 효과 → 감정 일관성 점수 계산까지 한 번에 수행한다.
"""
import cv2
from preprocessing_eq import preprocessing_image_eq
from preprocessing_st import preprocessing_image_st
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification, rotate_region, FACIAL_REGIONS
from cartoon import cartoon_effect
from test import emotion_consistency_score

UI_TO_REGION = {
    "눈":   ["left_eye", "right_eye"],
    "코":   ["nose"],
    "입":   ["mouth"],
    "대두": ["face_shape"],
}

def run_pipeline(
    image_path: str,
    parts: list[str],
    rotate_eyes: bool,
    mode: str = "eq",
    s: float = 0.8
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    if mode == "eq":
        processed = preprocessing_image_eq(img, s)
    else:
        processed = preprocessing_image_st(img, s)

    landmarks = extract_landmarks_mediapipe(img)

    modifications = {key: False for key in FACIAL_REGIONS}
    for part in parts:
        for region_key in UI_TO_REGION.get(part, []):
            modifications[region_key] = True

    modified = apply_modification(processed, landmarks, modifications)

    if rotate_eyes:
        modified = rotate_region(
            modified, landmarks,
            FACIAL_REGIONS["left_eye"],  90
        )
        modified = rotate_region(
            modified, landmarks,
            FACIAL_REGIONS["right_eye"], -90
        )

    cartooned = cartoon_effect(modified)

    score = emotion_consistency_score(img, cartooned)

    return cartooned, score