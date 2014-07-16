#!/usr/bin/env python
import sys
from PySide import QtGui, QtCore
import kernelWidget
import buffersWidget
import displayWidget
import newDialog
import darkorange_stylesheet
import settings
import icons
import pyGpuip as gpuip
from time import gmtime, strftime

class gpuip_gui(QtGui.QMainWindow):
    def __init__(self):
        super(gpuip_gui, self).__init__()
        self.setWindowTitle("gpuip gui")
        self.setWindowIcon(icons.get("pug"))

        # Start in center of the screen, covering 80% 
        r = QtGui.QDesktopWidget().availableGeometry()
        self.setGeometry(r.width()*0.10, r.height() * 0.10,
                         r.width() * 0.80, r.height() * 0.80)

        self.createMenuAndActions()
        self.createDockWidgets()
        
        # Central tab widget (main part of the gui)
        self.kernelTabWidget = QtGui.QTabWidget(self)
        self.setCentralWidget(self.kernelTabWidget)
                
        self.settings = None
        self.reset()
        
        if len(sys.argv) > 1:
            self.settings = settings.Settings()
            self.settings.read(sys.argv[1])
            self.initFromSettings()
            
    def new(self):
        dialog = newDialog.NewDialog(self)
        if dialog.exec_():
            self.settings = dialog.getSettings()
            self.initFromSettings()
            self.log("Creating a new session")

    def newFromExisting(self):
        f = QtGui.QFileDialog.getOpenFileName(
            None, "New from existing", QtCore.QDir.currentPath(), "ip (*ip)")
        if f[0]:
            s = settings.Settings()
            s.read(f[0])
            dialog = newDialog.NewDialog(self)
            dialog.initFromSettings(s)
            if dialog.exec_():
                self.settings = dialog.getSettings()
                self.initFromSettings()
                self.log("Creating new session from previous " + f[0])

    def open(self):
        f = QtGui.QFileDialog.getOpenFileName(
            None, "Open", QtCore.QDir.currentPath(), "ip (*ip)")
        if f[0]:
            self.settings = settings.Settings()
            self.settings.read(f[0])
            self.initFromSettings()
            self.log("Opening " + f[0])

    def save(self):
        f = QtGui.QFileDialog.getSaveFileName(
            None, "Save", QtCore.QDir.currentPath(), "ip (*ip)")
        if f[0]:
            self.updateSettings()
            self.settings.write(f[0])
            self.log("Saved current session to " + f[0])

    def updateSettings(self):
        # Get buffer input and outputs
        for b in self.settings.buffers:
            b.input = self.buffersWidget.getBufferInput(b.name)
            b.output = self.buffersWidget.getBufferOutput(b.name)

        # Get in buffers, out buffers and  param values
        for k in self.settings.kernels:
            kw = self.kernelWidgets[k.name]
            k.code = kw.codeEditor.toPlainText()

            for inb in k.inBuffers:
                inb.buffer = str(kw.inBuffers[inb.name].cbox.currentText())
            for outb in k.outBuffers:
                outb.buffer = str(kw.outBuffers[outb.name].cbox.currentText())
            for p in k.params:
                kernelParam = kw.params[p.name]
                p.value = eval(kernelParam.lineEdit.text())

    def initFromSettings(self):
        self.reset()

        self.gpuip, self.buffers, self.kernels = self.settings.create()

        bufferNames = [b.name for b in self.settings.buffers]
        for b in self.settings.buffers:
            self.buffersWidget.addBuffer(b.name, b.type, b.channels)
        self.buffersWidget.layout.addStretch()
        
        refresh = True
        self.kernelWidgets = {}
        for k in self.settings.kernels:
            w = kernelWidget.KernelWidget(self.kernelTabWidget)
            for inb in k.inBuffers:
                w.addInBuffer(inb.name, inb.buffer, bufferNames)
            for outb in k.outBuffers:
                w.addOutBuffer(outb.name, outb.buffer, bufferNames)
            for p in k.params:
                w.addParameter(p.name, p.value, p.default, p.min, p.max, p.type)
            self.kernelTabWidget.addTab(w, k.name)
            self.kernelWidgets[k.name] = w
            
            if k.code != "":
                w.codeEditor.setText(k.code)
                refresh = False
        if refresh:
            self.refreshBoilerplateCode(True)

    def reset(self):
        for i in range(self.kernelTabWidget.count()):
            self.kernelTabWidget.removeTab(0)

        self.logBrowser.clear()
        self.gpuip = None
        self.bufferData = None
        self.kernels = []
        self.buffers = {}
        self.kernelWidgets = {}
                
    def initBuffers(self):
        self.log("Initiating buffers...")

    def build(self):
        for kernel in self.kernels:
            kernelWidget = self.kernelWidgets[kernel.name]
            kernel.code = str(kernelWidget.codeEditor.toPlainText())
        
        self.log("Building kernels...")

        err = self.gpuip.Build()
        if err != "":
            self.logError(err)

    def process(self):
        self.log("Processing kernels...")

        self.displayWidget.refreshDisplay()

    def import_from_images(self):
        self.log("Importing data from input images...")
        print "import"

    def export_to_images(self):
        self.log("Exporting data to output images...")
        print "export"

    def run_all_steps(self):
        self.build()
        self.initBuffers()
        self.import_from_images()
        self.process()
        self.export_to_images()

    def log(self, msg):
        tt = str(strftime("[%Y-%m-%d %H:%M:%S]: ", gmtime()))
        self.logBrowser.append(tt + msg)

    def logError(self, msg):
        tt = str(strftime("[%Y-%m-%d %H:%M:%S] ", gmtime()))
        error = "<font color='red'>Error: </font>"
        self.logBrowser.append(tt + error + msg)

    def refreshBoilerplateCode(self, skipDialog = False):
        if not skipDialog:
            ret = QtGui.QMessageBox.warning(
                self, self.tr("Refresh Boilerplate Code"),
                self.tr("Refreshing the boilerplate code will remove previous"+\
                        " code. \nDo you want to continue?"),
                QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel,
                QtGui.QMessageBox.Cancel)
            if ret == QtGui.QMessageBox.StandardButton.Cancel:
                return

        for kernel in self.kernels:
            editor = self.kernelWidgets[kernel.name].codeEditor
            if skipDialog and str(editor.toPlainText()) != "":
                return
            code = self.gpuip.GetBoilerplateCode(kernel)
            editor.clear()
            editor.setText(code)

    def createDockWidgets(self):
        left = QtCore.Qt.LeftDockWidgetArea
        right = QtCore.Qt.RightDockWidgetArea
   
        # Create Log dock
        dock = QtGui.QDockWidget("Log", self)
        self.logBrowser = QtGui.QTextBrowser(dock)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        dock.setWidget(self.logBrowser)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.windowsMenu.addAction(dock.toggleViewAction())

        # Create buffers dock
        dockBuffers = QtGui.QDockWidget("Buffers", self)
        dockBuffers.setAllowedAreas(left | right)
        self.buffersWidget = buffersWidget.BuffersWidget(self)
        dockBuffers.setWidget(self.buffersWidget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dockBuffers)
        self.windowsMenu.addAction(dockBuffers.toggleViewAction())
        
        # Create display dock
        dock = QtGui.QDockWidget("Display", self)
        dock.setAllowedAreas(left | right)
        self.displayWidget = displayWidget.DisplayWidget(dock)
        dock.setWidget(self.displayWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.windowsMenu.addAction(dock.toggleViewAction())

        # The buffers tab starts with being stacked on the display dock
        self.tabifyDockWidget(dock, dockBuffers)

        # Create display debug dock (starts hidden)
        dock = QtGui.QDockWidget("Display Debug", self)
        dock.setAllowedAreas(left | right)
        self.displayDebugWidget = QtGui.QTextBrowser(dock)
        self.displayWidget.setDisplayDebug(self.displayDebugWidget)
        dock.setWidget(self.displayDebugWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.close()
        self.windowsMenu.addAction(dock.toggleViewAction())

    def createMenuAndActions(self):
        fileMenu = self.menuBar().addMenu("&File")
        toolBar = self.addToolBar("Toolbar")
        toolBar.setIconSize(toolBar.iconSize() * 1.5)

        newAction = QtGui.QAction(icons.get("new"), "&New",self)
        newAction.setShortcut(QtGui.QKeySequence.New)
        newAction.triggered.connect(self.new)
        fileMenu.addAction(newAction)
        toolBar.addAction(newAction)

        newExistingAction = QtGui.QAction(icons.get("newExisting"), 
                                          "&New from existing",self)
        newExistingAction.triggered.connect(self.newFromExisting)
        fileMenu.addAction(newExistingAction)
        toolBar.addAction(newExistingAction)

        openAction = QtGui.QAction(icons.get("open"), "&Open", self)
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.triggered.connect(self.open)
        fileMenu.addAction(openAction)
        toolBar.addAction(openAction)

        saveAction = QtGui.QAction(icons.get("save"),"&Save As", self)
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.triggered.connect(self.save)
        fileMenu.addAction(saveAction)
        toolBar.addAction(saveAction)

        fileMenu.addSeparator()
        quitAction = QtGui.QAction("&Quit", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.triggered.connect(self.close)
        fileMenu.addAction(quitAction)

        editorMenu = self.menuBar().addMenu("&Editor")
        toolBar.addSeparator()
        refreshAction = QtGui.QAction(icons.get("refresh"),
                                      "&Refresh boilerplate code",self)
        refreshAction.setShortcut("Ctrl+R")
        refreshAction.triggered.connect(self.refreshBoilerplateCode)
        editorMenu.addAction(refreshAction)
        toolBar.addAction(refreshAction)

        runMenu = self.menuBar().addMenu("&Run")
        toolBar.addSeparator()

        buildAction = QtGui.QAction(icons.get("build"), "1. &Build", self)
        buildAction.setShortcut("Ctrl+B")
        buildAction.triggered.connect(self.build)
        runMenu.addAction(buildAction)
        toolBar.addAction(buildAction)

        initAction = QtGui.QAction(icons.get("init"), "2. &Init Buffers", self)
        initAction.setShortcut("Ctrl+I")
        initAction.triggered.connect(self.initBuffers)
        runMenu.addAction(initAction)
        toolBar.addAction(initAction)

        importAction = QtGui.QAction(icons.get("import"),
                                     "3. &Import from images",self)
        importAction.setShortcut("Ctrl+Q")
        importAction.triggered.connect(self.import_from_images)
        runMenu.addAction(importAction)
        toolBar.addAction(importAction)

        processAction = QtGui.QAction(icons.get("process"), 
                                      "4. &Process", self)
        processAction.setShortcut("Ctrl+P")
        processAction.triggered.connect(self.process)
        runMenu.addAction(processAction)
        toolBar.addAction(processAction)

        exportAction = QtGui.QAction(icons.get("export"),
                                     "5. &Export to images",self)
        exportAction.setShortcut("Ctrl+W")
        exportAction.triggered.connect(self.export_to_images)
        runMenu.addAction(exportAction)
        toolBar.addAction(exportAction)

        runMenu.addSeparator()
        runAllAction = QtGui.QAction(QtGui.QIcon(""), "&All steps", self)
        runAllAction.setShortcut("Ctrl+A")
        runAllAction.triggered.connect(self.run_all_steps)
        runMenu.addAction(runAllAction)
        
        self.windowsMenu = self.menuBar().addMenu("&Windows")

        helpMenu = self.menuBar().addMenu("&Help")
        aboutQtAction = QtGui.QAction("About &Qt", self)
        aboutQtAction.triggered.connect(QtGui.qApp.aboutQt)
        helpMenu.addAction(aboutQtAction)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    mainwindow = gpuip_gui()
    mainwindow.setStyleSheet(darkorange_stylesheet.data)
    mainwindow.show()    
    sys.exit(app.exec_())
    
    
