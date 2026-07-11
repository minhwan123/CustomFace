import cv2
import numpy as np

def cartoon_effect(image):
    """엣지 검출(adaptive threshold)로 만든 윤곽선과 양방향 필터로 부드럽게
    처리한 색상을 합성해 만화(카툰) 스타일 이미지를 만든다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 10
    )
    color = cv2.bilateralFilter(image, 9, 300, 300)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_color)
    return cartoon
