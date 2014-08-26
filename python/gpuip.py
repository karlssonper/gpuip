#!/usr/bin/env python
import utils
import sys
import signal
import os
try:
    import argparse
    parsermodule = argparse.ArgumentParser
except:
    import optparse
    parsermodule = optparse.OptionParser
    parsermodule.add_argument = parsermodule.add_option

def getCommandLineArguments():
    # Command line arguments
    desc = "Framework for Image Processing on the GPU"
    parser = parsermodule(desc)
    parser.add_argument("-f", "--file",
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
    parser.add_argument("--timestamp",
                        action="store_true",
                        help="Add timestamp in log output")
    parser.add_argument("--nogui",
                        action="store_true",
                        help="Command line version")

    if parsermodule.__name__  == "ArgumentParser":
        return parser.parse_args()
    else:
        return parser.parse_args()[0]

def terminate(msg):
    print msg
    sys.exit(1)

def getSettings(args):
    import settings

    if not args.file or not os.path.isfile(args.file):
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
                param.setValue(utils.safeEval(value))
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
                if not os.path.isfile(buffer.input):
                    raise IOError("No such file: '%s'" % buffer.input)
            else:
                terminate("gpuip error: No buffer %s found." % buffer)

    # Change output buffers
    if args.outbuffer:
        for outb in args.outbuffer:
            bufferName, path = outb[0], outb[1]
            buffer = ipsettings.getBuffer(bufferName)
            if buffer:
                buffer.output = path
                os.makedirs(os.path.dirname(os.path.realpath(path)))
            else:
                terminate("gpuip error: No buffer %s found." % bufferName)

    return ipsettings

def runGUI(ippath, ipsettings):
    # Run GUI version
    from PySide import QtGui
    import mainwindow

    # Makes it possible to close program with ctrl+c in a terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
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
    ip, buffers, kernels = ipsettings.create()
    log("Created elements from settings.", overall_clock)

    ### 1. Build
    c = utils.StopWatch()
    check_error(ip.Build())
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
    ip.SetDimensions(width, height)
    check_error(ip.Allocate())
    log("Allocating done.", c)
    c = utils.StopWatch()
    for b in ipsettings.buffers:
        if b.input:
            check_error(ip.WriteBufferToGPU(buffers[b.name]))
    log("Transfering data to GPU done.", c)

    ### 4. Process
    c = utils.StopWatch()
    check_error(ip.Run())
    log("Processing done.", c)

    ### 5. Export buffers to images
    c = utils.StopWatch()
    for b in ipsettings.buffers:
        if b.output:
            log("Exporting data from %s to %s" %(b.name, b.output))
            check_error(ip.ReadBufferFromGPU(buffers[b.name]))
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

