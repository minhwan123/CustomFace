"""MediaPipe Face Mesh를 이용해 얼굴의 468개 랜드마크 좌표를 추출하는 모듈."""
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks_mediapipe(image):
    """image에서 얼굴 랜드마크 468개를 (x, y) 픽셀 좌표 리스트로 반환한다.
    얼굴이 검출되지 않으면 ValueError를 발생시킨다."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # 정밀 랜드마크 추출
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5  # 트래킹 신뢰도 추가
    ) as face_mesh:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected!")

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        # 정규화된 좌표 → 이미지 좌표로 변환
        points = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        return points  # (468개 좌표)

def get_landmark_coords(landmarks, indexes, image_shape):
    """landmarks 중 indexes에 해당하는 좌표들만 골라 numpy 배열로 반환한다."""
    h, w, _ = image_shape
    return np.array([[int(landmarks[i][0]), int(landmarks[i][1])] for i in indexes])
