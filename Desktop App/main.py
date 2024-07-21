from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication
from SchoolFinal1.frontPage import MySideBar
import sys
import cv2

from SchoolFinal1.login_FINAL import Ui_Dialog

#app = QApplication(sys.argv)
#window = MySideBar()
#window.show()
#app.exec()
# window = Ui_Dialog()
# window.show()
# app.exec()
app = QApplication(sys.argv)
# Create and show the login form
main_window = MySideBar()
main_window.LoginForm()
# Start the main event loop
sys.exit(app.exec())