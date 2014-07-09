from PySide import QtGui, QtCore

class BuffersWidget(QtGui.QWidget):
    class Buffer(object):
        def __init__(self, parent, gridLayout, row, name, format, channels):
            self.name = name
            self.inputfile = ""
            self.outputfile = ""
 
            nameLabel  = QtGui.QLabel(name, parent)
            formatLabel = QtGui.QLabel(format, parent)
            formatLabel.setAlignment(QtCore.Qt.AlignCenter)
            channelsLabel = QtGui.QLabel(str(channels), parent)
            channelsLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.inputLineEdit = QtGui.QLineEdit("", parent)
            self.outputLineEdit = QtGui.QLineEdit("", parent)
            inputButton = QtGui.QPushButton("...", parent)
            outputButton = QtGui.QPushButton("...", parent)
           
            gridLayout.addWidget(nameLabel, row + 1, 0)
            gridLayout.addWidget(formatLabel, row + 1, 2)
            gridLayout.addWidget(channelsLabel, row + 1, 4)
            gridLayout.addWidget(self.inputLineEdit, row + 1, 6)
            gridLayout.addWidget(inputButton, row + 1, 7)
            gridLayout.addWidget(self.outputLineEdit, row + 1, 9)
            gridLayout.addWidget(outputButton, row + 1, 10)
 
            inputButton.clicked.connect(self.selectInput)
            outputButton.clicked.connect(self.selectOutput)
 
        def selectInput(self):
             f = QtGui.QFileDialog.getOpenFileName(None,
                                                   "Select input image",
                                                   QtCore.QDir.currentPath(),
                                                   "Png (*png)")
             if f[0]:
                 self.inputLineEdit.setText(f[0])
 
        def selectOutput(self):
             f = QtGui.QFileDialog.getSaveFileName(None,
                                                   "Choose output image",
                                                   QtCore.QDir.currentPath(),
                                                   "Png (*png)")
             if f[0]:
                 self.outputLineEdit.setText(f[0])
                       
    def __init__(self, parent = None):
        super(BuffersWidget, self).__init__( parent)
 
        
        self.gridLayout = QtGui.QGridLayout()
        widget = QtGui.QWidget()
        widget.setLayout(self.gridLayout)

        vertLayout = QtGui.QVBoxLayout()
        vertLayout.addWidget(widget)
        self.setLayout(vertLayout)
        vertLayout.addStretch()
        
        placeHolderColumns = [1,3,5,8]
        for c in placeHolderColumns:
            self.gridLayout.setColumnMinimumWidth(c, 10)
        
        name = ["Name", "Format", "Channels", "Input", "Output"]
        pos = [0 , 2, 4, 6, 9]
        align = QtCore.Qt.AlignCenter
        for i in xrange(len(name)):
            label = QtGui.QLabel("<b>" + name[i] + "</b>", parent)
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(label, 0,  pos[i])

        self.buffers = {}
 
    def addBuffer(self, name, format, channels):
        self.buffers[name] = BuffersWidget.Buffer(
            self, self.gridLayout, len(self.buffers), name, format, channels)
