# run_app.py
from PyQt6.QtWidgets import QApplication
from main_ui import PoseEstimationUI

if __name__ == "__main__":
    app = QApplication([])
    window = PoseEstimationUI()
    window.show()
    app.exec()
