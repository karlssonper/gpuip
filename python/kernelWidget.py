from PySide import QtGui, QtCore
import sys

class KernelWidget(QtGui.QSplitter):
    def __init__(self, parent = None, callbackFunc = None):
        super(KernelWidget, self).__init__(QtCore.Qt.Horizontal,parent)
     
        #Optional callback function. Called everytime a paramter is changed
        self.callbackFunc = callbackFunc

        # Text Editor where the kernel code is displayed
        self.textEditor = QtGui.QPlainTextEdit(self)
        
        # To keep the grid compact, the group box widget has a vertical
        # layout with stretch added and a temporary widget inside with
        # a grid layout
        parameters = QtGui.QGroupBox("Parameters", self)
        vertLayout = QtGui.QVBoxLayout()
        parameters.setLayout(vertLayout)
        widget = QtGui.QWidget(parameters)
        vertLayout.addWidget(widget)
        self.paramsGridLayout = QtGui.QGridLayout()
        widget.setLayout(self.paramsGridLayout)
        vertLayout.addStretch()
        
        # Text and paramters is split vertically in a Horizontal box layout
        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.addWidget(self.textEditor)
        self.addWidget(parameters)
        
        self.params = {}

    def addParameter(self, name, defVal, minVal, maxVal, typename):
        self.params[name] = Parameter(self.paramsGridLayout, len(self.params),
                                      name, defVal, minVal, maxVal, 
                                      typename, self.callbackFunc)
        
class Parameter(object):
    def __init__(self, gridLayout, row, name, defVal, minVal, maxVal, typename,
                 callbackFunc = None):
        self.name = name
        self.defaultVal = defVal
        self.minVal = minVal
        self.maxVal = maxVal
        self.typename = typename
        self.callbackFunc = None

        # Each parameters has a label with the name, a lineedit with text value
        # and a slider with the value (relative to min max)
        self.label = QtGui.QLabel(name)
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setFixedWidth(40)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0,100)

        gridLayout.addWidget(self.label, row, 0)
        gridLayout.addWidget(self.lineEdit, row, 1)
        gridLayout.addWidget(self.slider, row, 2)
        
        # When a slider is changed it should update the line edit and vice verse 
        self.lineEdit.textChanged.connect(self.onLineEditChange)
        self.slider.valueChanged.connect(self.onSliderChange)
        
        # Helper variables to know when to trigger updates
        self.updateSlider = True
        self.updateLineEdit = True

        txt = str(int(defVal)) if typename == "int" else str(defVal)
        self.lineEdit.setText(txt)

    def onLineEditChange(self):
        # Changing the line edit triggers slider update that triggers 
        # line edit update again. This is to prevent the second update
        if not self.updateLineEdit:
            return

        # Evaluate the line edit text to get the number
        try:
            val = eval(self.lineEdit.text())
            if self.typename == "int":
                # If the parameter is of int, format line edit to be int too.
                self.lineEdit.setText(str(int(val)))
        except SyntaxError:
            # If error, fallback on the default value
            val = self.defaultVal
        
        # Don't run the onSliderChange  function
        self.updateSlider = False

        # Update the slider position
        if val < self.minVal:
            self.slider.setSliderPosition(0)
        elif val > self.maxVal:
            self.slider.setSliderPosition(100)
        else:
            t = (val - self.minVal) / float(self.maxVal - self.minVal)
            self.slider.setSliderPosition(100 * t)

        # Slider has been updated, safe to set this variable to true again
        self.updateSlider = True

        # If a callback function was added, call it
        if self.callbackFunc:
            self.callbackFunc()

    def onSliderChange(self):
        # Changing the slider triggers line edit update that triggers 
        # slider update again. This is to prevent the second update
        if not self.updateSlider:
            return 

        # Evaluate val based on slider position
        val = 0.01*self.slider.value() * (self.maxVal-self.minVal) + self.minVal

        # Don't run the onLineEditChange  function
        self.updateLineEdit = False

        # Update LineEdit text
        txt = str(int(val))if self.typename == "int" else str(val)
        self.lineEdit.setText(txt)

        # LineEdit has been updated, safe to set this variable to true again
        self.updateLineEdit = True

        # If a callback function was added, call it
        if self.callbackFunc:
            self.callbackFunc()
