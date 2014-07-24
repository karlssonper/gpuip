import kernelwidget
import bufferswidget
import displaywidget
import newdialog
import stylesheet
import settings
import icons
import utils
import pygpuip
from PySide import QtGui, QtCore
import sys
from time import gmtime, strftime

class MainWindow(QtGui.QMainWindow):
    def __init__(self, path, settings = None):
        super(MainWindow, self).__init__()
        self.path = path
        self.setWindowTitle("gpuip")
        self.setWindowIcon(icons.get("pug"))
        self.setStyleSheet(stylesheet.data)
        # Start in center of the screen, covering 80% 
        r = QtGui.QDesktopWidget().availableGeometry()
        self.setGeometry(r.width()*0.10, r.height() * 0.10,
                         r.width() * 0.80, r.height() * 0.80)

        self.toolbarIconSize = QtCore.QSize(32,32)
        self.interactive = False
        self.createMenuAndActions()
        self.createDockWidgets()
        
        # Central tab widget (main part of the gui)
        self.kernelTabWidget = QtGui.QTabWidget(self)
        self.setCentralWidget(self.kernelTabWidget)
        
        self.reset()

        self.settings = settings
        if self.settings:
            self.initFromSettings()
        else:
            self.new()

        self.needsBuild = True
        self.needsInitBuffers = True
        self.needsImport = True

    def new(self):
        dialog = newdialog.NewDialog(self)
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
            dialog = newdialog.NewDialog(self)
            dialog.initFromSettings(s)
            if dialog.exec_():
                self.settings = dialog.getSettings()
                self.initFromSettings()
                self.log("Creating new session from previous " + f[0])

    def open(self):
        f = QtGui.QFileDialog.getOpenFileName(
            self, "Open", QtCore.QDir.currentPath(), "ip (*ip)")
        if f[0]:
            self.settings = settings.Settings()
            self.settings.read(f[0])
            self.initFromSettings()
            self.log("Opening " + f[0])

    def save(self):
        self.updateSettings()
        self.settings.write(self.path)
        self.log("Saved current session to %s" % self.path)

    def saveAs(self):
        f = QtGui.QFileDialog.getSaveFileName(
            self, "Save",  QtCore.QDir.currentPath(), "ip (*ip)")
        if f[0]:
            self.path = f[0]
            self.save()

    def updateSettings(self):
        # Get buffer input and outputs
        for b in self.settings.buffers:
            b.input = self.buffersWidget.getBufferInput(b.name)
            b.output = self.buffersWidget.getBufferOutput(b.name)

        # Get in buffers, out buffers and  param values
        for k in self.settings.kernels:
            kw = self.kernelWidgets[k.name]
            k.code = str(kw.codeEditor.toPlainText())

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

        self.displayWidget.setBuffers(self.buffers)

        bufferNames = [b.name for b in self.settings.buffers]
        for b in self.settings.buffers:
            self.buffersWidget.addBuffer(b.name, b.type, 
                                         b.channels, b.input, b.output)
        self.buffersWidget.layout.addStretch()
        
        refresh = True
        self.kernelWidgets = {}
        for k in self.settings.kernels:
            w = kernelwidget.KernelWidget(self.kernelTabWidget, 
                                          self.interactiveProcess)
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
        self.logBrowser.clear()
        self.gpuip = None
        self.bufferData = None
        self.kernels = []
        self.buffers = {}
        self.kernelWidgets = {}

        # Re-add GUI components for buffers widget
        self.buffersWidget = bufferswidget.BuffersWidget(self)
        self.dockBuffers.setWidget(self.buffersWidget)

        # Remove all kernel widgets from the kernel tab widget
        for i in range(self.kernelTabWidget.count()):
            self.kernelTabWidget.removeTab(0)
                
    def initBuffers(self):
        self.updateSettings()
        bufferNames = ""
        inputBuffers = []
        for b in self.settings.buffers:
            if b.input:
                inputBuffers.append(b.input)
            bufferNames += b.name + ", "

        self.log("Initiating buffers [ <i> %s </i> ] ..." % bufferNames)
        width, height, err = utils.getLargestImageSize(inputBuffers)
        if err:
            self.logError(err)
            return False

        self.gpuip.SetDimensions(width, height)
        utils.allocateBufferData(self.buffers, width, height)
        err = self.gpuip.InitBuffers()
        if err:
            self.logError(err)
            return False
        else:
            self.log("Buffers were initiated successfully.")
            self.needsInitBuffers = False
            return True

    def build(self):
        kernelNames = ""
        for kernel in self.kernels:
            kernelWidget = self.kernelWidgets[kernel.name]
            kernel.code = str(kernelWidget.codeEditor.toPlainText())
            kernelNames += kernel.name + ", "
       
        self.log("Building kernels [ <i>%s</i> ] ..." % kernelNames[:-2])

        err = self.gpuip.Build()
        if not err:
            self.log("All kernels were built successfully.")
            self.needsBuild = False
            return True
        else:
            self.logError(err)
            QtGui.QMessageBox.critical(self, self.tr("Kernel Build Error"),
                               self.tr(err), QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)
            return False
            
    def interactiveProcess(self):
        if self.interactive:
            # Run previous steps if necessary. If any fails, return function
            if (self.needsBuild and not self.build()) or \
               (self.needsInitBuffers and not self.initBuffers()) or \
               (self.needsImport and not self.import_from_images()):
                return False
            self.process()

    def process(self):
        self.updateSettings()

        self.log("Processing kernels...")

        self.settings.updateKernels(self.kernels, self.buffers)
        err = self.gpuip.Process()
        if err:
            self.logError(err)
            return False

        self.log("All kernels were processed successfully...")

        for b in self.buffers:
            err = self.gpuip.ReadBuffer(self.buffers[b])
            if err:
                self.logError(err)
                return False

        self.displayWidget.refreshDisplay()

    def import_from_images(self):
        self.updateSettings()

        for b in self.settings.buffers:
            if b.input:
                self.log("Importing data from image <i>%s</i> to <i>%s</i>." \
                         % (b.input, b.name))
                err = utils.imgToNumpy(b.input, self.buffers[b.name].data)
                if err:
                    self.logError(err)
                    return False
                err = self.gpuip.WriteBuffer(self.buffers[b.name])
                if err:
                    self.logError(err)
                    return False
        self.log("Image data was successfully transfered to buffers.")

        for b in self.settings.buffers:
            if b.input:
                self.displayWidget.setActiveBuffer(b.name)
                self.needsImport = False
                return True
        return False

    def export_to_images(self):
        self.updateSettings()

        for b in self.settings.buffers:
            if b.output:
                self.log("Exporting data from buffer <i>%s</i> to <i>%s</i>." \
                         % (b.name, b.output))
                err = utils.numpyToImage(self.buffers[b.name].data, b.output)
                if err:
                    self.logError(err)
                    return False
                
        self.log("Buffer data was successfully transfered from to images.")
        return True

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

    def toggleInteractive(self):
        self.interactive = not self.interactive

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
        self.dockBuffers = QtGui.QDockWidget("Buffers", self)
        self.dockBuffers.setAllowedAreas(left | right)
        self.buffersWidget = bufferswidget.BuffersWidget(self)
        self.dockBuffers.setWidget(self.buffersWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dockBuffers)
        self.windowsMenu.addAction(self.dockBuffers.toggleViewAction())

        # Create display dock
        dock = QtGui.QDockWidget("Display", self)
        dock.setAllowedAreas(left | right)
        self.displayWidget = displaywidget.DisplayWidget(dock)
        checkBox = self.displayWidget.interactiveCheckBox
        checkBox.stateChanged.connect(self.toggleInteractive)
        dock.setWidget(self.displayWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.windowsMenu.addAction(dock.toggleViewAction())

        # The buffers tab starts with being stacked on the display dock
        self.tabifyDockWidget(dock, self.dockBuffers)

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
        menuNames = ["&File", "&Editor", "&Run", "&Windows", "&Help"]
        fileMenu, editorMenu, runMenu, self.windowsMenu, helpMenu = \
            [self.menuBar().addMenu(name) for name in menuNames]

        toolBar = self.addToolBar("Toolbar")
        toolBar.setIconSize(self.toolbarIconSize)
        
        # each item is a list of 
        # [icon, action name, shortcut, function, menu, add to toolbar]
        items = [
            [icons.get("new"), "&New", QtGui.QKeySequence.New, 
             self.new, fileMenu, True],

            [icons.get("newExisting"), "&New from existing", None, 
             self.newFromExisting, fileMenu, True],

            [icons.get("open"), "&Open", QtGui.QKeySequence.Open, 
             self.open, fileMenu, True],

            [icons.get("save"), "&Save", QtGui.QKeySequence.Save, 
             self.save, fileMenu, True],

            [icons.get("save"), "&Save As", QtGui.QKeySequence.SaveAs, 
             self.saveAs, fileMenu, False],

            [QtGui.QIcon(""), "&QuitQQQ", "Ctrl+Q", 
             self.close, fileMenu, False],

            [icons.get("refresh"), "&Refresh Boilerplate Code", "Ctrl+R", 
             self.refreshBoilerplateCode,editorMenu,True],

            [icons.get("build"), "1. &Build", "Ctrl+B", 
             self.build, runMenu, True],

            [icons.get("init"), "2. &Init Buffers", "Ctrl+I", 
             self.initBuffers, runMenu, True],

            [icons.get("import"), "3. &Import from images", "Ctrl+W", 
             self.import_from_images, runMenu, True],

            [icons.get("process"), "4. &Process", "Ctrl+P", 
             self.process, runMenu, True],

            [icons.get("export"), "5. &Export to images", "Ctrl+E", 
             self.export_to_images, runMenu, True],

            [QtGui.QIcon(""), "&All steps", "Ctrl+A", 
             self.run_all_steps, runMenu, False],

            [QtGui.QIcon(""), "About &Qt", None,
             QtGui.qApp.aboutQt, helpMenu, False]
            ]
            
        prevMenu = items[0][4] # to know when to add a separator in the toolbar
        for i in items:
            ico, name, shortcut, func, menu, add = i[0],i[1],i[2],i[3],i[4],i[5]

            # Create action
            action = QtGui.QAction(ico, name, self)
            
            # Set shortcut
            if shortcut:
                action.setShortcut(shortcut)
            
            # Connect the action with a function
            action.triggered.connect(func)

            # Add the action to a menu
            menu.addAction(action)

            # Add to toolbar
            if add:
                # If different menu then previous, add a separating line
                if menu != prevMenu:
                    toolBar.addSeparator()
                    prevMenu = menu
                toolBar.addAction(action)
           
        action = QtGui.QAction(QtGui.QIcon(""), "close", self)
        action.setShortcut("Ctrl+C")
