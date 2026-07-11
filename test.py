"""감정 일관성(emotion consistency) 점수 계산 모듈.

만화화(cartoonize)된 이미지가 원본 사진의 표정을 얼마나 잘 유지하는지를,
FER가 추출한 감정별 확률 벡터 간 코사인 유사도로 측정한다.
"""
import cv2
import numpy as np
from fer import FER
from sklearn.metrics.pairwise import cosine_similarity
import os

ORIGINAL_IMAGE_PATH = 'assets/origin/origin.png'
CARICATURE_IMAGE_PATH_EQ = 'assets/result/result_eq.png'
CARICATURE_IMAGE_PATH_ST = 'assets/result/result_st.png'


def get_emotion_vector(image):
    """FER로 image의 7가지 감정 확률 벡터를 추출한다.
    얼굴/감정을 검출하지 못하면 None을 반환한다."""
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(image)
    if not result:
        return None
    return np.array(list(result[0]['emotions'].values()))


def emotion_consistency_score(img1, img2):
    """두 이미지의 감정 벡터 간 코사인 유사도를 반환한다. (1.0에 가까울수록 표정이 유사)"""
    vec1 = get_emotion_vector(img1)
    vec2 = get_emotion_vector(img2)
    if vec1 is None or vec2 is None:
        print('감정 인식 실패')
        return None
    return cosine_similarity([vec1], [vec2])[0][0]


def main():
    if not (os.path.exists(ORIGINAL_IMAGE_PATH) and os.path.exists(CARICATURE_IMAGE_PATH_EQ) and os.path.exists(CARICATURE_IMAGE_PATH_ST)):
        print('평가용 이미지가 존재하지 않습니다. main.py를 먼저 실행하세요.')
        return None, None

    img1 = cv2.imread(ORIGINAL_IMAGE_PATH)
    img_eq = cv2.imread(CARICATURE_IMAGE_PATH_EQ)
    img_st = cv2.imread(CARICATURE_IMAGE_PATH_ST)

    if img1 is None or img_eq is None or img_st is None:
        print('이미지 로드 실패')
        return None, None

    score_eq = emotion_consistency_score(img1, img_eq)
    score_st = emotion_consistency_score(img1, img_st)
    print(f'Equalization Score: {score_eq:.4f}')
    print(f'Stretching Score: {score_st:.4f}')

    return score_eq, score_st

if __name__ == '__main__':
    main()
