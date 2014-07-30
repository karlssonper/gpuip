import numpy
from time import gmtime, strftime, time

class StopWatch(object):
    def __init__(self):
        self.t = time()
    
    def __str__(self):
        return "%.2f" %((time() - self.t) * 1000.0) + " ms"
        
def _exrToNumpy(filename, npyArray):
    try:
        import OpenEXR
        import Imath
    except Exception:
        return "Could not load input data from %s." % filename + \
            " No python support for OpenEXR."

    # Open EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get width and height
    dw = exr_file.header()['dataWindow']
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    npyWidth, npyHeight = npyArray.shape[0], npyArray.shape[1]
    if npyWidth != width or npyHeight != height:
        err = "Could not load input data from %s.\n" % filename
        err += "\tInput dimensions are %i x %i.\n" % (width, height)
        err += "\tgpuip dimensions are %i x %i.\n" % (npyWidth, npyHeight)
        return err

    # Read data from file
    imath_float = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [exr_file.channel(c, imath_float) for c in "RGB"]

    # Convert to 1D numpy arrays
    rgb = [numpy.fromstring(c,dtype=numpy.float32) for c in channels]

    # Copy from 1D arrays to the 3D
    for i in range(3):
        npyArray [:,:,i] = rgb[i].reshape(width,height)

    return ""

def _numpyToExr(npyArray, filename):
    try:
        import OpenEXR
        import Imath
    except Exception:
        return "Could not save data to %s." % filename + \
            " No python support for OpenEXR."

    HEADER = OpenEXR.Header(npyArray.shape[0],npyArray.shape[1])
    chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    HEADER['channels'] = dict([(c, chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(filename, HEADER)
    exr.writePixels({'R': npyArray[:,:,0].tostring(),
                     'G': npyArray[:,:,1].tostring(),
                     'B': npyArray[:,:,2].tostring()})
    exr.close()

    return ""

def _exrImageSize(filename):
    try:
        import OpenEXR
        import Imath
    except Exception:
        return -1, -1, "Could not load input data from %s." % filename + \
            " No python support for OpenEXR."

    # Open EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get width and height
    dw = exr_file.header()['dataWindow']

    # Return width, height
    return dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1, ""

def _pngToNumpy(filename, npyArray):
    try:
        from PIL import Image
    except Exception:
        return -1,-1, "Could not load input data from %s." % filename + \
            " No python support for PIL (Python Image Library)."
    image = Image.open(filename)
    for i in range(3):
        flat = numpy.array(image.getdata(i))
        npyArray[:,:,i] = flat.reshape(image.size[1],image.size[0])
    return ""

def _numpyToPng(npyArray, filename):
    try:
        from PIL import Image
    except Exception:
        return -1,-1, "Could not load input data from %s." % filename + \
            " No python support for PIL (Python Image Library)."
    array = numpy.zeros((npyArray.shape[0],npyArray.shape[1],3), npyArray.dtype)
    for i in xrange(3):
        array[:,:,i] = npyArray[:,:,i]
    out = Image.fromarray(array)
    out.save(filename)
    return ""

def _pngImageSize(filename):
    try:
        from PIL import Image
    except Exception:
        return 0,0, "Could not load input data from %s." % filename + \
            " No python support for PIL (Python Image Library)."
    image = Image.open(filename)
    return image.size[0], image.size[1], ""

def imgToNumpy(filename, numpy):
    lower = filename.lower()
    if lower.endswith("exr"):
        return _exrToNumpy(filename, numpy)
    elif lower.endswith("png"):
        return _pngToNumpy(filename, numpy)
    else:
        err = "Could not load input data from %s. " % filename
        err += "format '" + filename[filename.rfind("."):] + "' not supported."
        return err

def numpyToImage(numpy, filename):
    lower = filename.lower()
    if lower.endswith("exr"):
        return _numpyToExr(numpy, filename)
    elif lower.endswith("png"):
        return _numpyToPng(numpy, filename)
    else:
        err = "Could not export data to %s. " % filename
        err += "format '" + filename[filename.rfind("."):] + "' not supported."
        return err

def getImageSize(filename):
    lower = filename.lower()
    if lower.endswith("exr"):
        return _exrImageSize(filename)
    elif lower.endswith("png"):
        return _pngImageSize(filename)
    else:
        err = "Could not load input data from %s. " % filename
        err += "format '" + filename[filename.rfind("."):] + "' not supported."
        return -1, -1, err

def getLargestImageSize(filenames = []):
    maxw, maxh = 0, 0
    for filename in filenames:
        w, h, err = getImageSize(filename)
        if err == "" and maxw*maxh < w * h:
            maxw, maxh = w, h
        else:
            return maxw, maxh, err
    return maxw, maxh, ""

def allocateBufferData(buffers, width, height):
    for b in buffers:
        buf = buffers[b]
        ndtype = numpy.float32 if buf.bpp/buf.channels == 4 else numpy.ubyte
        channels = buf.channels if buf.channels != 3 else 3
        buf.data = numpy.zeros((width, height, channels), dtype = ndtype)

def getTimeStr():
    return str(strftime("[%Y-%m-%d %H:%M:%S] ", gmtime()))

