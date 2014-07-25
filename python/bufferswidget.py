from PySide import QtGui, QtCore

class BuffersWidget(QtGui.QWidget):
    class Buffer(object):
        def __init__(self, name, format, channels, input, output, parent):
            self.layout = QtGui.QVBoxLayout()

            self.inputLineEdit = QtGui.QLineEdit(input, parent)
            self.outputLineEdit = QtGui.QLineEdit(output, parent)
            self.inputLineEdit.setMinimumWidth(100)
            self.outputLineEdit.setMinimumWidth(99)
            inputButton = QtGui.QPushButton("...", parent)
            inputButton.clicked.connect(self.selectInput)
            outputButton = QtGui.QPushButton("...", parent)
            outputButton.clicked.connect(self.selectOutput)

            labelNames = ["Name", "Format", "Channels", "Input", "Output" ]
            rhsWidgets = [[QtGui.QLabel(name,parent)] , 
                          [QtGui.QLabel(format,parent)],
                          [QtGui.QLabel(str(channels),parent)],
                          [self.inputLineEdit, inputButton],
                          [self.outputLineEdit, outputButton]]

            for name, widgets in zip(labelNames, rhsWidgets):
                lhs = QtGui.QLabel("<b>"+name+": </b>", parent)
                layout = QtGui.QHBoxLayout()
                layout.addWidget(lhs)
                for widget in widgets:
                    layout.addWidget(widget)
                if len(widgets) == 1:
                    layout.addStretch()
                self.layout.addLayout(layout)
 
        def selectInput(self):
             inputImageFile = QtGui.QFileDialog.getOpenFileName(
                 None, "Select input image", 
                 QtCore.QDir.currentPath(), "Exr (*exr);;Png (*png)")
             if inputImageFile[0]:
                 self.inputLineEdit.setText(inputImageFile[0])
 
        def selectOutput(self):
            outputImageFile = QtGui.QFileDialog.getSaveFileName(
                 None, "Choose output image", 
                 QtCore.QDir.currentPath(), "Exr(*exr);;Png (*png)")
            if outputImageFile[0]:
                self.outputLineEdit.setText(outputImageFile[0])
                       
    def __init__(self, parent = None):
        super(BuffersWidget, self).__init__( parent)
 
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        self.buffers = {}
 
    def addBuffer(self, name, format, channels, input, output):
        self.buffers[name] = BuffersWidget.Buffer(name, format, channels, 
                                                  input, output, self)
        self.layout.addLayout(self.buffers[name].layout)

        # Add separating line after each buffer
        separator = QtGui.QFrame(self)
        separator.setFrameShape(QtGui.QFrame.HLine)
        separator.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout.addWidget(separator)

    def getBufferInput(self, name):
        return str(self.buffers[name].inputLineEdit.text())
 
    def getBufferOutput(self, name):
        return str(self.buffers[name].outputLineEdit.text())
     
    
