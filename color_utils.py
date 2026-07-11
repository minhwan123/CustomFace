"""히스토그램 평활화(equalization)와 히스토그램 스트레칭(stretching)
두 전처리 파이프라인(preprocessing_eq.py / preprocessing_st.py)이
공통으로 사용하는 색상 전처리 유틸리티 모듈.
동일한 로직이 두 파일에 중복되어 있던 것을 이 모듈로 통합했다.
"""
import cv2
import numpy as np


def histogram_equalization(y_channel: np.ndarray) -> np.ndarray:
    """단일 채널 이미지에 대해 전역 히스토그램 평활화를 적용한다."""
    hist, bins = np.histogram(y_channel.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])
    equalized = np.interp(y_channel.flatten(), bins[:-1], cdf_normalized)
    return equalized.reshape(y_channel.shape).astype(np.uint8)


def histogram_stretching(y_channel: np.ndarray) -> np.ndarray:
    """단일 채널 이미지의 밝기값을 [0, 255] 전체 범위로 선형 스트레칭한다."""
    min_val = np.min(y_channel)
    max_val = np.max(y_channel)
    stretched = (y_channel - min_val) * 255.0 / (max_val - min_val + 1e-8)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 ITU-R BT.601 기준 휘도(Y) 채널로 변환한다."""
    b, g, r = cv2.split(image)
    rgb = cv2.merge((r, g, b))
    coeffs = np.array([0.257, 0.504, 0.098])
    return np.tensordot(rgb, coeffs, axes=([-1], [0])) + 16


def color_preprocessing(image: np.ndarray, s: float, method: str = "stretch") -> np.ndarray:
    """지정한 방식으로 밝기를 정규화한 뒤, 원본 색상 비율을 s 제곱하여
    각 채널을 다시 합성한다. (s < 1이면 무채색에 가깝게, s > 1이면 채도를 강조)
    """
    b, g, r = cv2.split(image)
    y = rgb_to_y(image)

    if method == "equalize":
        y_adjusted = histogram_equalization(y)
    elif method == "stretch":
        y_adjusted = histogram_stretching(y)
    else:
        raise ValueError(f"알 수 없는 method입니다: {method!r} ('equalize' 또는 'stretch'만 지원)")

    brightness_factor = 1.15  # 전체 밝기를 15% 더 밝게 보정
    y_adjusted = np.clip(y_adjusted * brightness_factor, 0, 255).astype(np.uint8)

    r_out = np.clip(y_adjusted * np.power((r / (y + 1e-8)), s), 0, 255).astype(np.uint8)
    g_out = np.clip(y_adjusted * np.power((g / (y + 1e-8)), s), 0, 255).astype(np.uint8)
    b_out = np.clip(y_adjusted * np.power((b / (y + 1e-8)), s), 0, 255).astype(np.uint8)
    return cv2.merge((b_out, g_out, r_out))
