from PySide import QtGui, QtCore
import sys

class KernelWidget(QtGui.QSplitter):
    def __init__(self, parent = None, callbackFunc = None):
        super(KernelWidget, self).__init__(QtCore.Qt.Horizontal,parent)

        #Optional callback function. Called everytime a paramter is changed
        self.callbackFunc = callbackFunc

        # Left side is a Text Editor where the kernel code is displayed
        self.codeEditor = CodeEditor(self)

        # Right side is going to vertically placed group boxes
        # rightWidget is a work around since you can't add layouts to QSplitters
        rightWidget = QtGui.QWidget(self)
        rightLayout = QtGui.QVBoxLayout()
        rightWidget.setLayout(rightLayout)
        rightWidget.setSizePolicy(QtGui.QSizePolicy.Minimum,
                                  QtGui.QSizePolicy.Minimum)

        # Three group boxes for in buffers, out buffers and parameters
        groupBoxesNames = ["Input Buffers", "Output Buffers", "Parameters"]
        groupBoxes = [QtGui.QGroupBox(s,self) for s in groupBoxesNames]
        self.gridLayouts= {}
        for i,groupBox in enumerate(groupBoxes):
            self.gridLayouts[groupBoxesNames[i]] = QtGui.QGridLayout()
            groupBox.setLayout(self.gridLayouts[groupBoxesNames[i]])
            groupBox.setSizePolicy(QtGui.QSizePolicy.Minimum,
                                   QtGui.QSizePolicy.Minimum)
            rightLayout.addWidget(groupBox)
        rightLayout.addStretch()

        # Add widgets to splitter
        self.addWidget(self.codeEditor)
        self.addWidget(rightWidget)

        self.inBuffers = {}
        self.outBuffers = {}
        self.params = {}

    def addInBuffer(self, name, buffer, bufferNames):
        self.inBuffers[name] = Buffer(self, name, buffer, bufferNames,
                                      self.gridLayouts["Input Buffers"],
                                      len(self.inBuffers))

    def addOutBuffer(self, name, buffer, bufferNames):
        self.outBuffers[name] = Buffer(self, name, buffer, bufferNames,
                                      self.gridLayouts["Output Buffers"],
                                      len(self.outBuffers))

    def addParameter(self, name, val, defVal, minVal, maxVal, typename):
        self.params[name] = Parameter(self.gridLayouts["Parameters"],
                                      len(self.params), name, val, defVal,
                                      minVal, maxVal, typename,
                                      self.callbackFunc)
class Buffer(object):
    def __init__(self, parent, name, buffer, bufferNames, grid, row):
        self.name = name
        label = QtGui.QLabel(name+": ", parent)
        label.setSizePolicy(QtGui.QSizePolicy.Maximum,
                            QtGui.QSizePolicy.Maximum)
        self.cbox = QtGui.QComboBox(parent)
        self.cbox.addItems(bufferNames)
        if buffer != "":
            idx = bufferNames.index(buffer)
            if idx != -1:
                self.cbox.setCurrentIndex(idx)

        grid.addWidget(label, row, 0)
        grid.addWidget(self.cbox, row, 1)

class Parameter(object):
    def __init__(self, gridLayout, row, name, val, defVal, minVal, maxVal,
                 typename, callbackFunc = None):
        self.name = name
        self.defaultVal = defVal
        self.minVal = minVal
        self.maxVal = maxVal
        self.typename = typename
        self.callbackFunc = callbackFunc

        # Each parameters has a label with the name, a lineedit with text value
        # and a slider with the value (relative to min max)
        self.label = QtGui.QLabel(name)
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setFixedWidth(60)

        # Custom slider with a minimum width
        class _Slider(QtGui.QSlider):
            def __init__(self):
                super(_Slider, self).__init__(QtCore.Qt.Horizontal)
                self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,
                                   QtGui.QSizePolicy.Maximum)
            def sizeHint(self):
                return QtCore.QSize(80,20)
        self.slider = _Slider()
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

        txt = str(int(val)) if typename == "int" else str(val)
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

class CodeEditor(QtGui.QTextEdit):
    def __init__(self, parent):
        super(CodeEditor, self).__init__(parent)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        font.setFixedPitch(True);
        metrics = QtGui.QFontMetrics(font)
        self.setTabStopWidth(4 * metrics.width(' '))
        self.w = 55 * metrics.width(' ')
        self.setFont(font)
        color = QtGui.QColor(0,0,0)
        color.setNamedColor("#F8F8F2")
        self.setTextColor(color)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum,
                           QtGui.QSizePolicy.Minimum)
        self.highlighter = Highlighter(self.document())

    def sizeHint(self):
        return QtCore.QSize(self.w,200)

class Highlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(Highlighter, self).__init__(parent)

        keywordFormat = QtGui.QTextCharFormat()
        color = QtGui.QColor(0,0,0)
        color.setNamedColor("#66D9EF")
        keywordFormat.setForeground(color)

        keywords = ["char", "double",
                    "float", "float2", "float3", "float4",
                    "uchar", "uchar2", "uchar3", "uchar4",
                    "int", "int2", "int3", "int4",  "long",
                    "short", "signed", "unsigned", "union", "void"]
        keywordPatterns = ["\\b" + kw + "\\b" for kw in keywords]
        self.highlightingRules = [(QtCore.QRegExp(pattern), keywordFormat)
                for pattern in keywordPatterns]

        keywordFormat = QtGui.QTextCharFormat()
        color = QtGui.QColor(0,0,0)
        color.setNamedColor("#4e9a06")
        keywordFormat.setForeground(color)
        keywords = ["__kernel", "__global", "__global__"]
        keywordPatterns = ["\\b" + kw + "\\b" for kw in keywords]
        self.highlightingRules += [(QtCore.QRegExp(pattern), keywordFormat)
                for pattern in keywordPatterns]

        keywordFormat = QtGui.QTextCharFormat()
        color = QtGui.QColor(0,0,0)
        color.setNamedColor("#F92672")
        keywordFormat.setForeground(color)
        keywords = ["const", "inline", "template", "typedef", "typename",
                    "if", "for", "while", "switch", "case",
                    "return", "break", "else"]
        keywordPatterns = ["\\b" + kw + "\\b" for kw in keywords]
        self.highlightingRules += [(QtCore.QRegExp(pattern), keywordFormat)
                for pattern in keywordPatterns]

        color = QtGui.QColor(0,0,0)
        color.setNamedColor("#75715E")
        singleLineCommentFormat = QtGui.QTextCharFormat()
        singleLineCommentFormat.setForeground(color)
        self.highlightingRules.append((QtCore.QRegExp("//[^\n]*"),
                singleLineCommentFormat))

        self.multiLineCommentFormat = QtGui.QTextCharFormat()
        self.multiLineCommentFormat.setForeground(color)#QtCore.Qt.red)

        quotationFormat = QtGui.QTextCharFormat()
        quotationFormat.setForeground(QtCore.Qt.darkGreen)
        self.highlightingRules.append((QtCore.QRegExp("\".*\""),
                quotationFormat))

        self.commentStartExpression = QtCore.QRegExp("/\\*")
        self.commentEndExpression = QtCore.QRegExp("\\*/")

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QtCore.QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.indexIn(text)

        while startIndex >= 0:
            endIndex = self.commentEndExpression.indexIn(text, startIndex)

            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = len(text) - startIndex
            else:
                commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()

            self.setFormat(startIndex, commentLength,
                    self.multiLineCommentFormat)
            startIndex = self.commentStartExpression.indexIn(text,
                    startIndex + commentLength);

