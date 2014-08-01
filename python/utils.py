import numpy
from time import gmtime, strftime, time

class StopWatch(object):
    def __init__(self):
        self.t = time()
    
    def __str__(self):
        return "%.2f" %((time() - self.t) * 1000.0) + " ms"

def allocateBufferData(buffers):
    maxw, maxh = 0,0
    for bname in buffers:
        buf = buffers[bname]
        maxw = max(buf.data.shape[0], maxw)
        maxh = max(buf.data.shape[1], maxh)

    for bname in buffers:
        buf = buffers[bname]
        if buf.data.shape[0] != maxw or buf.data.shape[1] != maxh:
            ndtype = numpy.float32 if buf.bpp/buf.channels == 4 else numpy.ubyte
            channels = buf.channels if buf.channels != 3 else 3
            buf.data = numpy.zeros((maxw, maxh, channels), dtype = ndtype)

    return maxw, maxh
        
def getTimeStr():
    return str(strftime("[%Y-%m-%d %H:%M:%S] ", gmtime()))

