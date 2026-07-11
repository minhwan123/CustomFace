"""PyQt5 기반 GUI: 웹캠 실시간 미리보기/파일 업로드로 이미지를 입력받고,
체크박스로 선택한 부위를 pipeline.run_pipeline에 전달해 캐리커처 결과를 보여준다.
CLI 버전은 main.py를 참고.
"""
import os
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from pipeline import run_pipeline

def cv2_to_qpixmap(cv_img):
    h, w, ch = cv_img.shape
    bytes_line = ch * w
    qimg = QImage(cv_img.data, w, h, bytes_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

class CaricatureGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("캐리커처 생성기")
        self.setGeometry(100, 100, 820, 630)

        # ── 이미지 표시
        self.lbl_input  = QLabel(self); self.lbl_input.setGeometry(10, 10, 380, 380)
        self.lbl_output = QLabel(self); self.lbl_output.setGeometry(420, 10, 380, 380)

        # ── 입력 버튼
        btn_upload = QPushButton("파일 업로드", self)
        btn_upload.setGeometry(10, 410, 100, 30)
        btn_upload.clicked.connect(self.load_image)

        btn_capture = QPushButton("사진 촬영", self)
        btn_capture.setGeometry(120, 410, 100, 30)
        btn_capture.clicked.connect(self.capture_frame)

        # ── 수정 모드 토글 (체크박스)
        self.chk_eyes   = QCheckBox("눈 크게", self);  self.chk_eyes.move(10, 450)
        self.chk_nose   = QCheckBox("코 강조", self);  self.chk_nose.move(100,450)
        self.chk_mouth  = QCheckBox("입 강조", self);  self.chk_mouth.move(190,450)
        self.chk_head   = QCheckBox("대두 모드", self); self.chk_head.move(280,450)
        self.chk_rotate = QCheckBox("눈 회전", self);  self.chk_rotate.move(10, 490)

        # ── 실행 버튼
        btn_run = QPushButton("생성/적용", self)
        btn_run.setGeometry(10, 530, 100, 30)
        btn_run.clicked.connect(self.on_run)

        # ── 웹캠 스트리밍용 설정
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.setInterval(30)              
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        """30ms마다 호출되어 lbl_input에 웹캠 프레임을 띄워 줍니다."""
        ret, frame = self.cap.read()
        if not ret:
            return
        pix = cv2_to_qpixmap(frame).scaled(380,380)
        self.lbl_input.setPixmap(pix)
        self.current_frame = frame.copy()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        self.input_path = path
        pix = QPixmap(path).scaled(380,380)
        self.lbl_input.setPixmap(pix)

    def capture_frame(self):
        """실시간 스트리밍 중인 화면을 ‘촬영’하여 파일로 저장합니다."""
        if not hasattr(self, 'current_frame'):
            return
        tmp_path = "assets/origin/origin.png"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        cv2.imwrite(tmp_path, self.current_frame)
        self.input_path = tmp_path
        # input QLabel을 캡처한 프레임으로 고정
        pix = cv2_to_qpixmap(self.current_frame).scaled(380,380)
        self.lbl_input.setPixmap(pix)

    def on_run(self):
        parts = []
        if self.chk_eyes.isChecked():  parts.append("눈")
        if self.chk_nose.isChecked():  parts.append("코")
        if self.chk_mouth.isChecked(): parts.append("입")
        if self.chk_head.isChecked():  parts.append("대두")
        rotate = self.chk_rotate.isChecked()

        result_img, score = run_pipeline(self.input_path, parts, rotate)

        pix = cv2_to_qpixmap(result_img).scaled(380,380)
        self.lbl_output.setPixmap(pix)

        if score is None:
            print("감정 인식 실패: 점수를 계산할 수 없습니다.")
        else:
            print(f"Emotion Consistency Score: {score:.4f}")

    def closeEvent(self, event):
        """앱 종료 시 웹캠과 타이머 정리"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CaricatureGUI()
    win.show()
    sys.exit(app.exec_())