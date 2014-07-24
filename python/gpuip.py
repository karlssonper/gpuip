#!/usr/bin/env python
import settings
import utils
import argparse
import sys
import os

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
parser.add_argument("-ng", "--nogui", 
                    action="store_true",
                    help="Command line version")
args = parser.parse_args()

def terminate(msg):
    print msg
    sys.exit(1)

# Settings from *ip file
s = None
if os.path.isfile(args.file):
    s = settings.Settings()
    s.read(args.file)
    
    # Change parameter values
    if args.param:
        for p in args.param:
            kernelName, paramName, value = p
            kernel = s.getKernel(kernelName)
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
                buffer = s.getBuffer(bufferName)
                if buffer:
                    buffer.input = path
                else:
                    terminate("gpuip error: No buffer %s found." % buffer)

        # Change output buffers
        if args.outbuffer:
            for outb in args.outbuffer:
                bufferName, path = outb[0], outb[1]
                buffer = s.getBuffer(bufferName)
                if buffer:
                    buffer.output = path
                else:
                    terminate("gpuip error: No buffer %s found." % bufferName)

                
if not args.nogui:
    # Run GUI version
    import mainwindow

    # Makes it possible to close program with ctrl+c in a terminal
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
        
    from PySide import QtGui
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    mainwindow = mainwindow.MainWindow(path = args.file, settings = s)
    mainwindow.show()    
    sys.exit(app.exec_())
else:
    # Can't run non-gui version if there's no *.ip file
    if not s:
        err = "Must specify an existing *.ip file in the command-line version\n"
        err += "example: \n"
        err += "  gpuip --nogui smooth.ip"""
        terminate(err)
    
    def check_error(err):
        if err:
            terminate(err)
    
    ### 0. Create gpuip items from settings
    gpuip, buffers, kernels = s.create()

    ### 1. Build
    check_error(gpuip.Build())
    
    ### 2. Init Buffers
    bufferNames = ""
    inputBuffers = []
    for b in s.buffers:
        if b.input:
            inputBuffers.append(b.input)
        bufferNames += b.name + ", "
    width, height, error = utils.getLargestImageSize(inputBuffers)
    check_error(error)
    gpuip.SetDimensions(width, height)
    utils.allocateBufferData(buffers, width, height)
    check_error(gpuip.InitBuffers())
    
    ### 3. Import images to buffers
    for b in s.buffers:
        if b.input:
            check_error(utils.imgToNumpy(b.input, buffers[b.name].data))
            check_error(gpuip.WriteBuffer(buffers[b.name]))

    ### 4. Process
    check_error(gpuip.Process())
            
    ### 5. Export buffers to images
    for b in s.buffers:
        if b.output:
            check_error(gpuip.ReadBuffer(buffers[b.name]))
            check_error(utils.numpyToImage(buffers[b.name].data, b.output))
