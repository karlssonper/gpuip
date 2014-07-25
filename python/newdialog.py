from PySide import QtGui, QtCore
import settings

class NewDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(NewDialog, self).__init__(parent)

        r = QtGui.QDesktopWidget().availableGeometry()
        self.setGeometry(r.width()*0.25,
                         r.height() * 0.25,
                         r.width() * 0.5,
                         r.height() * 0.5)
      
        envGroupBox = QtGui.QGroupBox("Environment", self)
        layout = QtGui.QVBoxLayout()
        self.envComboBox = QtGui.QComboBox(envGroupBox)
        self.envComboBox.addItems(self.getEnvironments())
        layout.addWidget(self.envComboBox)
        envGroupBox.setLayout(layout)
        envGroupBox.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui.QSizePolicy.Maximum)

        self.buffersGroupBox = BufferGroupBox(self)
        self.kernels = []
                
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | \
                                               QtGui.QDialogButtonBox.Cancel)
        addKernelBtn = buttonBox.addButton("Add Kernel", 
                                           QtGui.QDialogButtonBox.ActionRole)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        addKernelBtn.clicked.connect(self.addKernel)


        self.topLayout = QtGui.QVBoxLayout()
        self.topLayout.addWidget(envGroupBox)
        self.topLayout.addWidget(self.buffersGroupBox)
        
        # Create tab widget
        self.tabWidget = QtGui.QTabWidget(self)
        tmp = QtGui.QWidget(self.tabWidget)
        tmp.setLayout(self.topLayout)
        self.tabWidget.addTab(tmp, "&General")

        self.addKernel()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.tabWidget)
        layout.addWidget(buttonBox)
        self.setLayout(layout)
        self.setWindowTitle("Configuration")

    @staticmethod
    def getEnvironments():
        return ["OpenCL", "CUDA", "GLSL"]

    def addKernel(self):
        kernel =  KernelGroupBox(len(self.kernels), self)
        self.kernels.append(kernel)
        self.tabWidget.addTab(kernel, "Kernel%i" % len(self.kernels))

    def initFromSettings(self, settings):
        idx = self.getEnvironments().index(settings.environment)
        self.envComboBox.setCurrentIndex(idx)

        for b in settings.buffers:
            self.buffersGroupBox.addBuffer()
            buffer = self.buffersGroupBox.buffers[-1]
            buffer.nameLineEdit.setText(b.name)
            idx = ["float", "uchar"].index(b.type)
            buffer.typeComboBox.setCurrentIndex(idx)
            buffer.channelsComboBox.setCurrentIndex(b.channels-1)

        for i,k in enumerate(settings.kernels):
            if i: # First kernel is always added automatically
                self.addKernel()
            
            kernel = self.kernels[-1]
            kernel.nameLineEdit.setText(k.name)
            kernel.codeFileLineEdit.setText(k.code_file)
            
            # In Buffers
            kernel.inBuffersComboBox.setCurrentIndex(len(k.inBuffers))
            for lineEdit, inb in zip(kernel.inBufferLineEdits, k.inBuffers):
                lineEdit.setText(inb.name)

            # Out Buffers
            kernel.outBuffersComboBox.setCurrentIndex(len(k.outBuffers))
            for lineEdit, outb in zip(kernel.outBufferLineEdits, k.outBuffers):
                lineEdit.setText(outb.name)

            for p in k.params:
                param = kernel.addParam()
                param.nameLineEdit.setText(p.name)
                idx = ["float", "int"].index(p.type)
                param.typeComboBox.setCurrentIndex(idx)
                param.defaultLineEdit.setText(str(p.default))
                param.minLineEdit.setText(str(p.min))
                param.maxLineEdit.setText(str(p.max))

    def getSettings(self):
        s = settings.Settings()
        
        # Environment
        s.environment = str(self.envComboBox.currentText())
        
        # Buffers
        for b in self.buffersGroupBox.buffers:
            s.buffers.append(settings.Settings.Buffer(
                    str(b.nameLineEdit.text()),
                    str(b.typeComboBox.currentText()).strip(),
                    eval(str(b.channelsComboBox.currentText()))))

        # Kernels
        for k in self.kernels:
            sk = settings.Settings.Kernel(
                str(k.nameLineEdit.text()).replace(" ", "/"),
                str(k.codeFileLineEdit.text()))

            # In Buffers
            for le in k.inBufferLineEdits:
                sk.inBuffers.append(
                    settings.Settings.Kernel.KernelBuffer(str(le.text()),""))

            # Out Buffers
            for le in k.outBufferLineEdits:
                sk.outBuffers.append(
                    settings.Settings.Kernel.KernelBuffer(str(le.text()),""))

            # Params
            for p in k.params:
                sk.params.append(settings.Settings.Param(
                        str(p.nameLineEdit.text()),
                        str(p.typeComboBox.currentText()).strip(),
                        eval(p.defaultLineEdit.text()),
                        eval(p.minLineEdit.text()),
                        eval(p.maxLineEdit.text())))
            s.kernels.append(sk)

        return s
       
class BufferGroupBox(QtGui.QGroupBox):
    class Buffer(object):
        def __init__(self, gridLayout, row, parent):
            self.nameLineEdit = QtGui.QLineEdit("buffer%i" % row, parent)
            self.typeComboBox = QtGui.QComboBox(parent)
            self.typeComboBox.addItems(["float\t","uchar\t"])
            self.channelsComboBox = QtGui.QComboBox(parent)
            self.channelsComboBox.addItems(["1", "2", "3", "4"])

            gridLayout.addWidget(self.nameLineEdit, row, 0)
            gridLayout.addWidget(self.typeComboBox, row, 1)
            gridLayout.addWidget(self.channelsComboBox, row, 2)
            
    def __init__(self, parent):
         super(BufferGroupBox,self).__init__(parent)
    
         self.buffers = []

         self.gridLayout = QtGui.QGridLayout()
         self.gridLayout.addWidget(QtGui.QLabel("Buffer Name",self), 0, 0)
         self.gridLayout.addWidget(QtGui.QLabel("Type",self), 0, 1)
         self.gridLayout.addWidget(QtGui.QLabel("Channels",self), 0, 2)

         addBufferButton = QtGui.QPushButton("    Add Buffer    ", self)
         addBufferButton.clicked.connect(self.addBuffer)
         addBufferButton.setSizePolicy(QtGui.QSizePolicy.Maximum,
                                       QtGui.QSizePolicy.Maximum)

         layout = QtGui.QVBoxLayout()
         layout.addLayout(self.gridLayout)
         layout.addWidget(addBufferButton)
         self.setLayout(layout)
         layout.addStretch()

    def addBuffer(self):
        b = BufferGroupBox.Buffer(self.gridLayout, len(self.buffers)+1, self)
        self.buffers.append(b)
         
class KernelGroupBox(QtGui.QGroupBox):
    class Parameter(object):
        def __init__(self, gridLayout, row, parent):
            self.nameLineEdit = QtGui.QLineEdit("param%i" % row, parent)
            self.typeComboBox = QtGui.QComboBox(parent)
            self.typeComboBox.addItems(["float\t","int\t"])
            self.defaultLineEdit = QtGui.QLineEdit("0.0",parent)
            self.minLineEdit = QtGui.QLineEdit("0.0",parent)
            self.maxLineEdit = QtGui.QLineEdit("10.0",parent)

            gridLayout.addWidget(self.nameLineEdit, row, 0)
            gridLayout.addWidget(self.typeComboBox, row, 1)
            gridLayout.addWidget(self.defaultLineEdit, row, 2)
            gridLayout.addWidget(self.minLineEdit, row, 3)
            gridLayout.addWidget(self.maxLineEdit, row, 4)
            
        def getData(self):
            name = self.nameLineEdit.text()
            type = str(self.typeComboBox.currentText()).strip()
            defaultVal = self.defaultLineEdit.text()
            minVal = self.minLineEdit.text()
            maxVal = self.maxLineEdit.text()

            if type == "int":
                defaultVal = str(int(eval(defaultVal)))
                minVal = str(int(eval(minVal)))
                maxVal = str(int(eval(maxVal)))
                
            return name, type, defaultVal, minVal, maxVal 

    def __init__(self, number, parent):
        super(KernelGroupBox, self).__init__("Kernel %i" % number, parent)
        self.parentDialog = parent

        nameLabel = QtGui.QLabel("Name: ", self)
        self.nameLineEdit = QtGui.QLineEdit("Untitled%i" % number, self)
        topLayout = QtGui.QHBoxLayout()
        topLayout.addWidget(nameLabel)
        topLayout.addWidget(self.nameLineEdit)
        
        codeFileLabel = QtGui.QLabel("Code: ", self)
        self.codeFileLineEdit = QtGui.QLineEdit("")
        codeBtn = QtGui.QPushButton("...")
        codeBtn.clicked.connect(self.onCodeBtnPress)
        midLayout = QtGui.QHBoxLayout()
        midLayout.addWidget(codeFileLabel)
        midLayout.addWidget(self.codeFileLineEdit)
        midLayout.addWidget(codeBtn)

        inBuffersGroupBox = QtGui.QGroupBox("Input Buffers", self)
        self.inBuffersLayout = QtGui.QVBoxLayout()
        inBuffersGroupBox.setLayout(self.inBuffersLayout)
        layout = QtGui.QHBoxLayout()
        label = QtGui.QLabel("Number of input buffers:", inBuffersGroupBox)
        label.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.inBuffersComboBox = QtGui.QComboBox(inBuffersGroupBox)
        self.inBuffersComboBox.addItems([str(i) for i in range(6)])
        self.inBuffersComboBox.currentIndexChanged.connect(self.setInBuffers)
        layout.addWidget(label)
        layout.addWidget(self.inBuffersComboBox)
        self.inBufferLineEdits = []
        self.inBuffersLayout.addLayout(layout)
        self.inBuffersLayout.addStretch()
        
        outBuffersGroupBox = QtGui.QGroupBox("Output Buffers", self)
        self.outBuffersLayout = QtGui.QVBoxLayout()
        outBuffersGroupBox.setLayout(self.outBuffersLayout)
        layout = QtGui.QHBoxLayout()
        label = QtGui.QLabel("Number of output buffers:", outBuffersGroupBox)
        label.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.outBuffersComboBox = QtGui.QComboBox(outBuffersGroupBox)
        self.outBuffersComboBox.addItems([str(i) for i in range(6)])
        self.outBuffersComboBox.currentIndexChanged.connect(self.setOutBuffers)
        layout.addWidget(label)
        layout.addWidget(self.outBuffersComboBox)
        self.outBufferLineEdits = []
        self.outBuffersLayout.addLayout(layout)
        self.outBuffersLayout.addStretch()
        
        paramGroupBox = QtGui.QGroupBox("Parameters", self)
        
        self.gridLayout = QtGui.QGridLayout()
        paramGroupBox.setLayout(self.gridLayout)
        self.gridLayout.addWidget(QtGui.QLabel("Name",self), 0, 0)
        self.gridLayout.addWidget(QtGui.QLabel("Type",self), 0, 1)
        self.gridLayout.addWidget(QtGui.QLabel("Default",self), 0, 2)
        self.gridLayout.addWidget(QtGui.QLabel("Min",self), 0, 3)
        self.gridLayout.addWidget(QtGui.QLabel("Max",self), 0, 4)
        
        self.params = []

        addParamButton = QtGui.QPushButton("    Add Param    ", self)
        addParamButton.clicked.connect(self.addParam)
        addParamButton.setSizePolicy(QtGui.QSizePolicy.Maximum,
                                     QtGui.QSizePolicy.Maximum)
        
        # Shrink all group boxes
        for groupBox in [inBuffersGroupBox, outBuffersGroupBox, paramGroupBox]:
            groupBox.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Maximum)
        layout = QtGui.QVBoxLayout()
        layout.addLayout(topLayout)
        layout.addLayout(midLayout)
        layout.addWidget(inBuffersGroupBox)
        layout.addWidget(outBuffersGroupBox)
        layout.addWidget(paramGroupBox)
        layout.addWidget(addParamButton)
        layout.addStretch()
        self.setLayout(layout)

    def addInBuffer(self, name):
        lineEdit = QtGui.QLineEdit(name)
        self.inBuffersLayout.addWidget(lineEdit)
        self.inBufferLineEdits.append(lineEdit)

    def addOutBuffer(self, name):
        lineEdit = QtGui.QLineEdit(name)
        self.outBuffersLayout.addWidget(lineEdit)
        self.outBufferLineEdits.append(lineEdit)
        
    def setInBuffers(self, idx):
        for lineEdit in self.inBufferLineEdits:
            self.inBuffersLayout.removeWidget(lineEdit)
        self.inBufferLineEdits = []
        for i in range(idx):
            self.addInBuffer("inBuffer%i" %i)

    def setOutBuffers(self, idx):
        for lineEdit in self.outBufferLineEdits:
            self.outBuffersLayout.removeWidget(lineEdit)
        self.outBufferLineEdits = []
        for i in range(idx):
            self.addOutBuffer("outBuffer%i" %i)

    def onCodeBtnPress(self):
        suffix = ".cl"
        if str(self.parentDialog.envComboBox.currentText()) == "CUDA":
            suffix = ".cu"
        elif str(self.parentDialog.envComboBox.currentText()) == "GLSL":
            suffix = ".glsl"
        name = "/" + self.getName() + suffix
        title = "Code output file for kernel " + self.getName()
        f = QtGui.QFileDialog.getSaveFileName(
            None, title,  QtCore.QDir.currentPath() + name, " (*)")
        if f[0]:
            self.codeFileLineEdit.setText(f[0])

    def getName(self):
        return self.nameLineEdit.text()

    def addParam(self):
        p = KernelGroupBox.Parameter(self.gridLayout, len(self.params)+1, self)
        self.params.append(p)
        return p
       
