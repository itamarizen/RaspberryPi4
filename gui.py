import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess
# import pdb;pdb.set_trace()
# add base directory to sys.path
#base_dir = r'C:\Users\User\RaspberryPi4_PCfolder\repo_learningopenCV\chapter10'
base_dir = os.getcwd()
sys.path.append(os.path.abspath(base_dir))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set a modern color palette
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#f6f6f6'))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor('#f6f6f6'))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor('#333'))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor('#66B2FF'))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor('#fff'))
        self.setPalette(palette)

        # Set the window title and size
        self.acceptDrops()
        self.setWindowTitle("Image Processing Application")
        
        # setting  the geometry of window
        self.setGeometry(50,50,1400,900)
        #self.setMinimumSize(400, 300)
        #self.setMaximumSize(1200, 900)
        self.setWindowIcon(QtGui.QIcon('logo.jpg'))

        # Use layout to create whitespace
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)

        # Use a modern font for text
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.setFont(font)

        # Use a high-quality image for the logo
        logo = QtGui.QPixmap("logo.jpg")
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setPixmap(logo.scaled(100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        layout.addWidget(self.logo_label, 0, QtCore.Qt.AlignHCenter)

        # Use a responsive layout
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label, 1)
        layout.addStretch()

        # Use an animation to highlight the start button
        self.start_button = QtWidgets.QPushButton("faces_dnn")
        self.start_button.setStyleSheet("QPushButton { background-color: #66B2FF; color: #fff; border-radius: 5px; padding: 10px 20px; } QPushButton:hover { background-color: #4D94FF; }")
        self.start_button.clicked.connect(self.faces_dnn)
        layout.addWidget(self.start_button, 0, QtCore.Qt.AlignCenter)


        # Use an animation to highlight the start button
        self.start_button = QtWidgets.QPushButton("objects_dnn")
        self.start_button.setStyleSheet("QPushButton { background-color: #66B2FF; color: #fff; border-radius: 5px; padding: 10px 20px; } QPushButton:hover { background-color: #4D94FF; }")
        self.start_button.clicked.connect(self.objects_dnn)
        layout.addWidget(self.start_button, 0, QtCore.Qt.AlignCenter)        

        # Use a consistent layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def resizeEvent(self, event):
        # get the current size of the window
        window_size = self.size()

        # set the size and position of the logo label
        logo_size = int(window_size.width() * 0.25)
        self.logo_label.setGeometry(QtCore.QRect(20, 20, logo_size, logo_size))

        # set the scaled pixmap to the logo label
        pixmap = QtGui.QPixmap("logo.jpg")
        pixmap = pixmap.scaled(logo_size, logo_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)

    def faces_dnn(self):
        # Code to start processing images from cameras and detecting obstacles goes here
        # run the Python script using the subprocess module
        script_path = os.path.join(base_dir, 'faces_dnn.py')
        subprocess.Popen(["python", script_path],cwd=base_dir)
        pass


    def objects_dnn(self):
        # Code to start processing images from cameras and detecting obstacles goes here
        # run the Python script using the subprocess module
        script_path = os.path.join(base_dir, 'objects_dnn.py')
        subprocess.Popen(["python", script_path],cwd=base_dir)
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
