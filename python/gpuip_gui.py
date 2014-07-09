#!/usr/bin/env python
import sys
from PySide import QtGui, QtCore
import kernelWidget
import buffersWidget
import displayWidget
import darkorange_stylesheet

# temp function
def createKernelWidget(parent):
    w = kernelWidget.KernelWidget(parent)
    w.addParameter("test1", 0, -10, 10, "int")
    w.addParameter("test2", 0, -10, 10, "int")
    w.addParameter("test3", 0, -10, 10, "float")
    return w

class gpuip_gui(QtGui.QMainWindow):
    def __init__(self):
        super(gpuip_gui, self).__init__()
        self.setWindowTitle("gpuip gui")
        self.setWindowIcon(QtGui.QIcon('pug.png'))

        # Create actions and menu items
        fileMenu = self.menuBar().addMenu("&File")
        
        newAction = QtGui.QAction(QtGui.QIcon(""), "&New", self)
        newAction.setShortcut(QtGui.QKeySequence.New)
        newAction.triggered.connect(self.new)
        fileMenu.addAction(newAction)

        newExistingAction = QtGui.QAction(QtGui.QIcon(""), 
                                          "&New from existing", self)
        newExistingAction.triggered.connect(self.newFromExisting)
        fileMenu.addAction(newExistingAction)

        openAction = QtGui.QAction(QtGui.QIcon(""), "&Open", self)
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.triggered.connect(self.open)
        fileMenu.addAction(openAction)

        saveAction = QtGui.QAction(QtGui.QIcon(""), "&Save As", self)
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.triggered.connect(self.save)
        fileMenu.addAction(saveAction)

        fileMenu.addSeparator()
        quitAction = QtGui.QAction("&Quit", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.triggered.connect(self.close)
        fileMenu.addAction(quitAction)

        runMenu = self.menuBar().addMenu("&Run")

        initAction = QtGui.QAction(QtGui.QIcon(""), "&Init Buffers", self)
        initAction.setShortcut("Ctrl+I")
        initAction.triggered.connect(self.initBuffers)
        runMenu.addAction(initAction)

        buildAction = QtGui.QAction(QtGui.QIcon(""), "&Build", self)
        buildAction.setShortcut("Ctrl+B")
        buildAction.triggered.connect(self.build)
        runMenu.addAction(buildAction)

        importAction = QtGui.QAction(QtGui.QIcon(""),"&Import from images",self)
        importAction.setShortcut("Ctrl+Q")
        importAction.triggered.connect(self.import_from_images)
        runMenu.addAction(importAction)

        processAction = QtGui.QAction(QtGui.QIcon(""), "&Process", self)
        processAction.setShortcut("Ctrl+P")
        processAction.triggered.connect(self.process)
        runMenu.addAction(processAction)

        exportAction = QtGui.QAction(QtGui.QIcon(""), "&Export to images",self)
        exportAction.setShortcut("Ctrl+W")
        exportAction.triggered.connect(self.export_to_images)
        runMenu.addAction(exportAction)

        runMenu.addSeparator()
        runAllAction = QtGui.QAction(QtGui.QIcon(""), "&All steps", self)
        runAllAction.setShortcut("Ctrl+R")
        runAllAction.triggered.connect(self.run_all_steps)
        runMenu.addAction(runAllAction)
        
        windowsMenu = self.menuBar().addMenu("&Windows")

        helpMenu = self.menuBar().addMenu("&Help")
        aboutQtAction = QtGui.QAction("About &Qt", self)
        aboutQtAction.triggered.connect(QtGui.qApp.aboutQt)
        helpMenu.addAction(aboutQtAction)

        # Central tab widget (main part of the gui)
        tabWidget = QtGui.QTabWidget(self)
        self.setCentralWidget(tabWidget)

        # Temp populate
        b = buffersWidget.BuffersWidget(tabWidget)
        b.addBuffer("test", "uchar", 4)
        b.addBuffer("test123", "float", 1)
        tabWidget.addTab(b, "&Buffers")
        tabs = [createKernelWidget(tabWidget) for i in range(3)]
        for i in range(3):
            tabWidget.addTab(tabs[i], "&Tab%s" % i)
           
        # Create Log dock
        dock = QtGui.QDockWidget("Log", self)
        self.log = QtGui.QTextBrowser(dock)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        windowsMenu.addAction(dock.toggleViewAction())

        dock = QtGui.QDockWidget("Display", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        dWidget = displayWidget.DisplayWidget(dock)
        dock.setWidget(dWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        windowsMenu.addAction(dock.toggleViewAction())
        
    def new(self):
        print "new"

    def newFromExisting(self):
        print "new from existing"

    def open(self):
        print "open"

    def save(self):
        print "save"
    
    def initBuffers(self):
        print "init"

    def build(self):
        print "build"

    def process(self):
        print "process"

    def import_from_images(self):
        print "import"

    def export_to_images(self):
        print "export"

    def run_all_steps(self):
        self.initBuffers()
        self.build()
        self.import_from_images()
        self.process()
        self.export_to_images()
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    mainwindow = gpuip_gui()
    mainwindow.setStyleSheet(darkorange_stylesheet.data)
    mainwindow.show()    
    sys.exit(app.exec_())
