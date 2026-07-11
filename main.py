"""CLI 진입점: 웹캠 캡처 → 얼굴 랜드마크 추출 → 부위별 왜곡 →
카툰 효과 → 감정 일관성 점수 비교까지 전체 파이프라인을 순서대로 실행한다.
GUI 버전은 app.py를 참고.
"""
from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification, rotate_region, FACIAL_REGIONS
from cartoon import cartoon_effect
from preprocessing_eq import preprocessing_image_eq
from preprocessing_st import preprocessing_image_st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import test  # 별도 프로세스로 띄우지 않고 점수 계산 함수를 직접 import해서 사용

def main():
    img = capture_face_with_guidelines()
    if img is None:
        print("⚠️ 캡처 실패 또는 취소되었습니다.")
        return

    modifications, rotate_eyes = get_user_modifications()
    landmarks = extract_landmarks_mediapipe(img)
    processed_img_eq = preprocessing_image_eq(img, 0.5)
    processed_img_st = preprocessing_image_st(img, 0.5)
    modified_img_eq = apply_modification(processed_img_eq, landmarks, modifications)
    modified_img_st = apply_modification(processed_img_st, landmarks, modifications)

    if rotate_eyes:
        modified_img_eq = rotate_region(modified_img_eq, landmarks, FACIAL_REGIONS["left_eye"], 90)
        modified_img_eq = rotate_region(modified_img_eq, landmarks, FACIAL_REGIONS["right_eye"], -90)
        modified_img_st = rotate_region(modified_img_st, landmarks, FACIAL_REGIONS["left_eye"], 90)
        modified_img_st = rotate_region(modified_img_st, landmarks, FACIAL_REGIONS["right_eye"], -90)

    cartooned_eq = cartoon_effect(modified_img_eq)
    cartooned_st = cartoon_effect(modified_img_st)

    origin_dir = 'assets/origin'
    result_dir = 'assets/result'
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    origin_path = os.path.join(origin_dir, 'origin.png')
    result_path_eq = os.path.join(result_dir, 'result_eq.png')
    result_path_st = os.path.join(result_dir, 'result_st.png')

    cv2.imwrite(origin_path, img)
    cv2.imwrite(result_path_eq, cartooned_eq)
    cv2.imwrite(result_path_st, cartooned_st)

    # 감정 유사도 점수 계산
    score_eq, score_st = test.main()
    print(f"Equalization Score: {score_eq}, Stretching Score: {score_st}")

    # 더 높은 점수를 받은 이미지를 출력
    if score_eq is not None and score_st is not None:
        if score_eq >= score_st:
            best_img = cartooned_eq
            best_title = "Best: Histogram Equalization"
        else:
            best_img = cartooned_st
            best_title = "Best: Histogram Stretching"

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB))
        plt.title(best_title)
        plt.axis('off')
        plt.show()
    else:
        print("⚠️ 감정 점수를 비교할 수 없습니다.")

if __name__ == "__main__":
    main()
