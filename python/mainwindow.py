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
            self.logSuccess("All buffers were initiated.")
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
            self.logSuccess("All kernels were built.")
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

        self.logSuccess("All kernels were processed.")

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
        self.logSuccess("Image data transfered to all buffers.")

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

        self.logSuccess("Buffer data transfered to images.")
        return True

    def run_all_steps(self):
        self.build()
        self.initBuffers()
        self.import_from_images()
        self.process()
        self.export_to_images()

    def log(self, msg):
        self.logBrowser.append(utils.getTimeStr() + msg)

    def logSuccess(self, msg):
        success = "<font color='green'>Success: </font>"
        self.logBrowser.append(utils.getTimeStr() + success + msg)

    def logError(self, msg):
        error = "<font color='red'>Error: </font>"
        self.logBrowser.append(utils.getTimeStr() + error + msg)

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
        LEFT = QtCore.Qt.LeftDockWidgetArea
        RIGHT = QtCore.Qt.RightDockWidgetArea

        # Create Log dock
        dock = QtGui.QDockWidget("Log", self)
        self.logBrowser = QtGui.QTextBrowser(dock)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        dock.setWidget(self.logBrowser)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.windowsMenu.addAction(dock.toggleViewAction())

        # Create buffers dock
        self.dockBuffers = QtGui.QDockWidget("Buffers", self)
        self.dockBuffers.setAllowedAreas(LEFT | RIGHT)
        self.buffersWidget = bufferswidget.BuffersWidget(self)
        self.dockBuffers.setWidget(self.buffersWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dockBuffers)
        self.windowsMenu.addAction(self.dockBuffers.toggleViewAction())

        # Create display dock
        dock = QtGui.QDockWidget("Display", self)
        dock.setAllowedAreas(LEFT | RIGHT)
        self.displayWidget = displaywidget.DisplayWidget(dock)
        checkBox = self.displayWidget.interactiveCheckBox
        checkBox.stateChanged.connect(self.toggleInteractive)
        dock.setWidget(self.displayWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.windowsMenu.addAction(dock.toggleViewAction())

        # The buffers tab starts with being stacked on the display dock
        self.tabifyDockWidget(dock, self.dockBuffers)

    def createMenuAndActions(self):
        menuNames = ["&File", "&Editor", "&Run", "&Windows", "&Help"]
        fileMenu, editorMenu, runMenu, self.windowsMenu, helpMenu = \
            [self.menuBar().addMenu(name) for name in menuNames]

        toolBar = self.addToolBar("Toolbar")
        toolBar.setIconSize(self.toolbarIconSize)

        def _addAction(icon, actionName, shortcut, func, menu, toolbar):
            action = QtGui.QAction(icon, actionName, self)
            action.triggered.connect(func)
            if shortcut:
                action.setShortcut(shortcut)
            menu.addAction(action)
            if toolbar:
                toolbar.addAction(action)

        _addAction(icons.get("new"), "&New", QtGui.QKeySequence.New,
                   self.new, fileMenu, toolBar)
        _addAction(icons.get("newExisting"), "&New from existing", None,
                   self.newFromExisting, fileMenu, toolBar),
        _addAction(icons.get("open"), "&Open", QtGui.QKeySequence.Open,
                   self.open, fileMenu, toolBar),
        _addAction(icons.get("save"), "&Save", QtGui.QKeySequence.Save,
                   self.save, fileMenu, toolBar),
        _addAction(icons.get("save"), "&Save As", QtGui.QKeySequence.SaveAs,
                   self.saveAs, fileMenu, None),
        _addAction(QtGui.QIcon(""), "&Quit", "Ctrl+Q",
                   self.close, fileMenu, None),
        toolBar.addSeparator()
        _addAction(icons.get("refresh"), "&Refresh Boilerplate Code", "Ctrl+R",
                   self.refreshBoilerplateCode,editorMenu,toolBar),
        toolBar.addSeparator()
        _addAction(icons.get("build"), "1. &Build", "Ctrl+B",
                   self.build, runMenu, toolBar),
        _addAction(icons.get("init"), "2. &Init Buffers", "Ctrl+I",
                   self.initBuffers, runMenu, toolBar),
        _addAction(icons.get("import"), "3. &Import from images", "Ctrl+W",
                   self.import_from_images, runMenu, toolBar),
        _addAction(icons.get("process"), "4. &Process", "Ctrl+P",
                   self.process, runMenu, toolBar),
        _addAction(icons.get("export"), "5. &Export to images", "Ctrl+E",
                   self.export_to_images, runMenu, toolBar),
        _addAction(QtGui.QIcon(""), "&All steps", "Ctrl+A",
                   self.run_all_steps, runMenu, None),

        _addAction(QtGui.QIcon(""), "About &Qt", None,
                   QtGui.qApp.aboutQt, helpMenu, None)

