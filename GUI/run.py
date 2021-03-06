# pyqt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit, QLineEdit, QComboBox, QRadioButton, QLabel, QGraphicsView
from PyQt5 import uic
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap
# spark model
from spark_model import *
# matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
# other
import sys


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("main.ui", self)

        # find the widgets in the xml file
        # Pattern
        self.line_edit_id = self.findChild(QLineEdit, "line_edit_id")
        self.combo_year = self.findChild(QComboBox, "combo_year")
        self.combo_month = self.findChild(QComboBox, "combo_month")
        self.combo_day = self.findChild(QComboBox, "combo_day")
        self.combo_pattern = self.findChild(QComboBox, "combo_pattern")
        self.button_show_pattern = self.findChild(
            QPushButton, "button_show_pattern")

        # Model
        self.radio_this = self.findChild(QRadioButton, "radio_this")
        self.radio_all = self.findChild(QRadioButton, "radio_all")
        self.combo_clustering = self.findChild(QComboBox, "combo_clustering")
        self.combo_classification = self.findChild(
            QComboBox, "combo_classification")
        self.button_run_model = self.findChild(QPushButton, "button_run_model")

        # Result
        self.label_image = self.findChild(QLabel, "label_image")
        self.label_acc = self.findChild(QLabel, "label_acc")
        self.label_auroc = self.findChild(QLabel, "label_auroc")
        self.label_aupr = self.findChild(QLabel, "label_aupr")

        # Action
        self.button_show_pattern.clicked.connect(self.show_pattern)
        self.button_run_model.clicked.connect(self.run_model)

        self.show()

    def show_pattern(self):
        customer_id = self.line_edit_id.text()
        year = self.combo_year.currentText()
        month = self.combo_month.currentText()
        day = self.combo_day.currentText()
        pattern = self.combo_pattern.currentText()
        date = year+"-"+month+"-"+day
        print("ID: " + customer_id + " Date: " + date + " Pattern: " + pattern)
        # get daily usage
        usage = collect_daily_vector(customer_id, date)
        # call plot function
        plot_malicious_samples(
            usage, [True, True, True, True, True, True, True])
        pixmap = QPixmap("attack.png")
        self.label_image.setPixmap(pixmap)

    def run_model(self):
        this_cutomer = self.radio_this.isChecked()
        all_cutomer = self.radio_all.isChecked()
        clustering = self.combo_clustering.currentText()
        classification = self.combo_classification.currentText()
        print("This: " + str(this_cutomer) + " All: " +
              str(all_cutomer) + " Clustering: " + clustering + " Classification: " + classification)


app = QApplication(sys.argv)
window = UI()
app.exec_()
