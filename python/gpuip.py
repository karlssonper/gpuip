#!/usr/bin/env python
import settings
import utils
import argparse
import sys
import signal
import os

def getCommandLineArguments():
    # Command line arguments
    desc = "Framework for Image Processing on the GPU"
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("file",
                        help="Image Processing file *.ip")
    parser.add_argument("-p", "--param",
                        action="append",
                        nargs = 3,
                        metavar = ("kernel", "param", "value"),
                        help="Change value of a parameter.")
    parser.add_argument("-i", "--inbuffer",
                        action="append",
                        nargs = 2,
                        metavar = ("buffer", "path"),
                        help = "Set input image to a buffer")
    parser.add_argument("-o", "--outbuffer",
                        action="append",
                        nargs = 2,
                        metavar = ("buffer", "path"),
                        help = "Set output image to a buffer")
    parser.add_argument("-v","--verbose",
                        action="store_true",
                        help="Outputs information")
    parser.add_argument("-ts","--timestamp",
                        action="store_true",
                        help="Add timestamp in log output")
    parser.add_argument("-ng", "--nogui",
                        action="store_true",
                        help="Command line version")
    return parser.parse_args()

def terminate(msg):
    print msg
    sys.exit(1)

def getSettings(args):
    if not os.path.isfile(args.file):
        return None

    ipsettings = settings.Settings()
    ipsettings.read(args.file)

    # Change parameter values
    if args.param:
        for p in args.param:
            kernelName, paramName, value = p
            kernel = ipsettings.getKernel(kernelName)
            if not kernel:
                terminate("gpuip error: No kernel %s found." % kernelName)
            param = kernel.getParam(paramName)
            if param:
                param.setValue(eval(value))
            else:
                terminate("gpuip error: No param %s found in kernel %s." \
                          % (paramName, kernelName))

        # Change input buffers
        if args.inbuffer:
            for inb in args.inbuffer:
                bufferName, path = inb[0], inb[1]
                buffer = ipsettings.getBuffer(bufferName)
                if buffer:
                    buffer.input = path
                else:
                    terminate("gpuip error: No buffer %s found." % buffer)

        # Change output buffers
        if args.outbuffer:
            for outb in args.outbuffer:
                bufferName, path = outb[0], outb[1]
                buffer = ipsettings.getBuffer(bufferName)
                if buffer:
                    buffer.output = path
                else:
                    terminate("gpuip error: No buffer %s found." % bufferName)

    return ipsettings

def runGUI(ippath, ipsettings):
    # Run GUI version
    import mainwindow

    # Makes it possible to close program with ctrl+c in a terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PySide import QtGui
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    mainwindow = mainwindow.MainWindow(path = ippath, settings = ipsettings)
    mainwindow.show()
    sys.exit(app.exec_())

def runCommandLine(ipsettings, verbose):
    # Can't run non-gui version if there's no *.ip file
    if not ipsettings:
        err = "Must specify an existing *.ip file in the command-line version\n"
        err += "example: \n"
        err += "  gpuip --nogui smooth.ip"""
        terminate(err)

    def check_error(err):
        if err:
            terminate(err)

    def log(text, stopwatch = None, time = True):
        time = time and args.timestamp
        if verbose:
            stopwatchStr = str(stopwatch) if stopwatch else  ""
            timeStr = utils.getTimeStr() if time else ""
            print timeStr + text + " " + stopwatchStr

    overall_clock = utils.StopWatch()

    ### 0. Create gpuip items from settings
    gpuip, buffers, kernels = ipsettings.create()
    log("Created elements from settings.", overall_clock)

    ### 1. Build
    c = utils.StopWatch()
    check_error(gpuip.Build())
    log("Building kernels [%s]." %  [k.name for k in kernels], c)
    
    ### 2. Import data from images
    c = utils.StopWatch()
    for b in ipsettings.buffers:
        if b.input:
            log("Importing data from %s to %s" %(b.input, b.name))
            check_error(buffers[b.name].Read(b.input, utils.getNumCores()))
    log("Importing data done.", c)
            
    ### 3. Allocate and transfer data to GPU
    c = utils.StopWatch()
    width, height = utils.allocateBufferData(buffers)
    gpuip.SetDimensions(width, height)
    check_error(gpuip.Allocate())
    log("Allocating done.", c)
    c = utils.StopWatch()
    for b in ipsettings.buffers:
        if b.input:
            check_error(gpuip.WriteBufferToGPU(buffers[b.name]))
    log("Transfering data to GPU done.", c)

    ### 4. Process
    c = utils.StopWatch()
    check_error(gpuip.Process())
    log("Processing done.", c)

    ### 5. Export buffers to images
    c = utils.StopWatch()
    for b in ipsettings.buffers:
        if b.output:
            log("Exporting data from %s to %s" %(b.name, b.output))
            check_error(gpuip.ReadBufferFromGPU(buffers[b.name]))
            check_error(buffers[b.name].Write(b.output,utils.getNumCores()))
    log("Exporting data done.", c)

    log("\nAll steps done. Total runtime:", overall_clock, time = False)

if __name__ == "__main__":
    args = getCommandLineArguments()
    ipsettings = getSettings(args)
    if args.nogui:
        runCommandLine(ipsettings, args.verbose)
    else:
        runGUI(args.file, ipsettings)

