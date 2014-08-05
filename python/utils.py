import pygpuip
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
            if buf.type == pygpuip.BufferType.UNSIGNED_BYTE:
                ndtype = numpy.ubyte
            elif buf.type == pygpuip.BufferType.HALF:
                ndtype = numpy.float16
            elif buf.type == pygpuip.BufferType.FLOAT:
                ndtype = numpy.float32
            elif buf.type == pygpuip.BufferType.DOUBLE:
                ndtype = numpy.float64
            buf.data = numpy.zeros((maxw, maxh, buf.channels), dtype = ndtype)

    return maxw, maxh
        
def getNumCores():
    import multiprocessing
    return multiprocessing.cpu_count()

def getTimeStr():
    return str(strftime("[%Y-%m-%d %H:%M:%S] ", gmtime()))

