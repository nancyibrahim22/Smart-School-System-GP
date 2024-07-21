# -*- coding: utf-8 -*-
import sys

################################################################################
## Form generated from reading UI file 'loginDialog24June.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
                               QLineEdit, QPushButton, QSizePolicy, QWidget, QMessageBox)
import resources_rc
#from SchoolFinal1.frontPage import MySideBar


class Ui_Dialog(QDialog):
    # def __init__(self):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(535, 560)
        self.setStyleSheet(u"background-color: #ffffff;")
        self.loginLbl = QLabel(self)
        self.loginLbl.setObjectName(u"loginLbl")
        self.loginLbl.setGeometry(QRect(270, 10, 111, 51))
        font = QFont()
        font.setPointSize(25)
        font.setBold(True)
        self.loginLbl.setFont(font)
        self.loginLbl.setStyleSheet(u"color: rgb(14, 13, 106);\n"
"")
        self.password_lbl = QLabel(self)
        self.password_lbl.setObjectName(u"password_lbl")
        self.password_lbl.setGeometry(QRect(30, 220, 121, 41))
        font1 = QFont()
        font1.setPointSize(20)
        self.password_lbl.setFont(font1)
        self.password_lbl.setStyleSheet(u"color: rgb(14, 13, 106);\n"
"")
        self.line = QFrame(self)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(0, 90, 621, 16))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.email_lbl = QLabel(self)
        self.email_lbl.setObjectName(u"email_lbl")
        self.email_lbl.setGeometry(QRect(40, 110, 71, 41))
        self.email_lbl.setFont(font1)
        self.email_lbl.setStyleSheet(u"color: rgb(14, 13, 106);\n"
"")
        self.email_lineEdit = QLineEdit(self)
        self.email_lineEdit.setObjectName(u"email_lineEdit")
        self.email_lineEdit.setGeometry(QRect(230, 120, 191, 31))
        self.email_lineEdit.setStyleSheet(u"QLineEdit{\n"
"	padding-left:15px;\n"
"	border: 1px solid #bbbbbc;\n"
"	border-radius: 15px;\n"
"	color: #0e0d6a;\n"
"	/*background-color: rgb(187, 187, 188);*/\n"
"}")

        self.password_lineEdit = QLineEdit(self)
        self.password_lineEdit.setObjectName(u"password_lineEdit")
        self.password_lineEdit.setGeometry(QRect(230, 230, 191, 31))
        self.password_lineEdit.setStyleSheet(u"QLineEdit{\n"
"	padding-left:15px;\n"
"	border: 1px solid #bbbbbc;\n"
"	border-radius: 15px;\n"
"	color: #0e0d6a;\n"
"	/*background-color: rgb(187, 187, 188);*/\n"
"}")
        self.password_lineEdit.setEchoMode(QLineEdit.Password)
        self.loginBtn = QPushButton(self)
        self.loginBtn.setObjectName(u"loginBtn")
        self.loginBtn.setGeometry(QRect(270, 320, 111, 41))
        font2 = QFont()
        font2.setBold(True)
        self.loginBtn.setFont(font2)
        self.loginBtn.setStyleSheet(u"QPushButton{\n"
"	background-color: #0192ef;\n"
"	color: #ffffff;\n"
"	border-radius: 5px;\n"
"	border: none;\n"
"	font-weight: bold;\n"
"	font-size: 15px;\n"
"}")
        self.label = QLabel(self)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(110, 0, 141, 91))
        self.label.setPixmap(QPixmap(u":/Icons/schoolLogo2.jpg"))
        self.label.setScaledContents(True)

        self.retranslateUi(self)

        QMetaObject.connectSlotsByName(self)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.loginLbl.setText(QCoreApplication.translate("Dialog", u"Login", None))
        self.password_lbl.setText(QCoreApplication.translate("Dialog", u"Password", None))
        self.email_lbl.setText(QCoreApplication.translate("Dialog", u"Email", None))
        self.loginBtn.setText(QCoreApplication.translate("Dialog", u"Login", None))
        self.label.setText("")
    # retranslateUi

