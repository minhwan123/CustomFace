"""색상 전처리 단계 중 히스토그램 평활화(equalization) 방식."""
from color_utils import color_preprocessing


def preprocessing_image_eq(image, s: float = 0.5):
    """히스토그램 평활화 기반으로 밝기/색상을 정규화한다.

    s는 채도(색상 강조) 정도를 조절한다: 1보다 작으면 무채색에 가깝게,
    1보다 크면 색이 더 진하게 표현된다.
    """
    return color_preprocessing(image, s, method="equalize")
