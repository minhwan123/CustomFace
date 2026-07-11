import cv2
import mediapipe as mp


def capture_face_with_guidelines():
    """웹캠을 열어 얼굴이 검출된 프레임을 기다린다.

    얼굴이 검출된 상태에서 아무 키나 누르면 캡처, Esc를 누르면 취소한다.
    캡처된 BGR 프레임을 반환하며, 취소되었거나 카메라 프레임이 없으면 None을 반환한다.
    """
    cap = cv2.VideoCapture(0)
    captured = None
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_detected = bool(results.detections)

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)
        if key != -1 and face_detected:  # 얼굴이 보이는 상태에서 아무 키나 누르면 캡처
            captured = frame.copy()
            break
        elif key == 27:  # Esc 키는 취소
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return captured
