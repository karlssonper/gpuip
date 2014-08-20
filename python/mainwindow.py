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
        self.needsAllocate = True
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
                p.value = utils.safeEval(kernelParam.lineEdit.text())

    def initFromSettings(self):
        self.reset()

        self.ip, self.buffers, self.kernels = self.settings.create()

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
        self.ip = None
        self.bufferData = None
        self.kernels = []
        self.buffers = {}
        self.kernelWidgets = {}

        # Re-add GUI components for buffers widget
        scroll = QtGui.QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.buffersWidget = bufferswidget.BuffersWidget(scroll)
        scroll.setWidget(self.buffersWidget)
        self.dockBuffers.setWidget(scroll)
        self.buffersWidget.show()

        # Remove all kernel widgets from the kernel tab widget
        for i in range(self.kernelTabWidget.count()):
            self.kernelTabWidget.removeTab(0)

    def build(self):
        kernelNames = ""
        for kernel in self.kernels:
            kernelWidget = self.kernelWidgets[kernel.name]
            kernel.code = str(kernelWidget.codeEditor.toPlainText())
            kernelNames += kernel.name + ", "

        self.log("Building kernels [ <i>%s</i> ] ..." % kernelNames[:-2])

        clock = utils.StopWatch()
        err = self.ip.Build()
        if not err:
            self.logSuccess("All kernels were built.", clock)
            self.needsBuild = False
            return True
        else:
            self.logError(err)
            QtGui.QMessageBox.critical(self, self.tr("Kernel Build Error"),
                               self.tr(err), QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)
            return False

    
    def import_from_images(self):
        self.updateSettings()

        clock = utils.StopWatch()
        for b in self.settings.buffers:
            if b.input:
                self.log("Importing data from image <i>%s</i> to <i>%s</i>." \
                         % (b.input, b.name))
                err = self.buffers[b.name].Read(b.input, utils.getNumCores())
                if err:
                    self.logError(err)
                    return False
        self.logSuccess("Image data imported", clock)
        self.displayWidget.refreshDisplay()
        self.needsImport = False
        return True

    def allocate(self):
        self.updateSettings()

        clock = utils.StopWatch()
        bufferNames = [b.name for b in self.settings.buffers]
        self.log("Allocating buffers <i> %s </i> ..." % bufferNames)
        width, height = utils.allocateBufferData(self.buffers)
        self.ip.SetDimensions(width, height)
        err = self.ip.Allocate()
        clock = utils.StopWatch()
        if err:
            self.logError(err)
            return False
        else:
            self.logSuccess("All buffers were allocated.", clock)
            self.needsAllocate = False

        clock = utils.StopWatch()
        for b in self.settings.buffers:
            if b.input:
                err = self.ip.WriteBufferToGPU(self.buffers[b.name])
                if err:
                    self.logError(err)
                    return False
        self.logSuccess("Data transfered to GPU.", clock)
        return True

    def interactiveProcess(self):
        if self.interactive:
            # Run previous steps if necessary. If any fails, return function
            if (self.needsBuild and not self.build()) or \
               (self.needsAllocate and not self.allocate()) or \
               (self.needsImport and not self.import_from_images()):
                return False
            self.run()

    def run(self):
        self.updateSettings()

        self.log("Running kernels...")

        self.settings.updateKernels(self.kernels, self.buffers)
        clock = utils.StopWatch()
        err = self.ip.Run()
        if err:
            self.logError(err)
            return False

        self.logSuccess("All kernels processed.", clock)

        clock = utils.StopWatch()
        for b in self.buffers:
            err = self.ip.ReadBufferFromGPU(self.buffers[b])
            if err:
                self.logError(err)
                return False
        self.logSuccess("Data transfered from GPU.", clock)
        self.displayWidget.refreshDisplay()
        return True

    def export_to_images(self):
        self.updateSettings()

        clock = utils.StopWatch()
        for b in self.settings.buffers:
            if b.output:
                self.log("Exporting data from buffer <i>%s</i> to <i>%s</i>." \
                         % (b.name, b.output))
                err = self.buffers[b.name].Write(b.output, utils.getNumCores())
                if err:
                    self.logError(err)
                    return False

        self.logSuccess("Buffer data transfered to images.", clock)
        return True

    def run_all_steps(self):
        for f in ["build","import_from_images","allocate","process","export_to_images"]:
            getattr(self,f)() # run func
            QtGui.QApplication.instance().processEvents() # update gui
        return True

    def log(self, msg):
        self.logBrowser.append(utils.getTimeStr() + msg)

    def logSuccess(self, msg, clock):
        success = "<font color='green'>Success: </font>"
        clockStr= "<i> " + str(clock) + "</i>"
        self.logBrowser.append(utils.getTimeStr() + success + msg + clockStr)

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
            code = self.ip.GetBoilerplateCode(kernel)
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
        _addAction(icons.get("import"), "2. &Import from images", "Ctrl+W",
                   self.import_from_images, runMenu, toolBar),
        _addAction(icons.get("init"), "3. &Allocate", "Ctrl+I",
                   self.allocate, runMenu, toolBar),
        _addAction(icons.get("process"), "4. &Run", "Ctrl+P",
                   self.run, runMenu, toolBar),
        _addAction(icons.get("export"), "5. &Export to images", "Ctrl+E",
                   self.export_to_images, runMenu, toolBar),
        _addAction(QtGui.QIcon(""), "&All steps", "Ctrl+A",
                   self.run_all_steps, runMenu, None),

        _addAction(QtGui.QIcon(""), "About &Qt", None,
                   QtGui.qApp.aboutQt, helpMenu, None)

