import os
import glob
import sys
import numpy
# no need to generate again
if os.path.exists(sys.argv[1]):
    sys.exit(0)
try:
    from PIL import Image
except:
    txt = ("raise Exception('gpuip: Icons were not generated. "
           "Install the Python module PIL and rebuild.')")
    open(sys.argv[1], "w").write(txt)
    sys.exit(0)

icons = {}
for i in glob.glob("*.png"):
    icons[i[:i.find(".")]] = i

out = """
from PySide import QtGui
data = {}
width = {}
height = {}
def get(name):
    image = QtGui.QImage(data[name], width[name], height[name], 
                         QtGui.QImage.Format_ARGB32)
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))
"""

for i in icons:
    im = Image.open(icons[i])
    im = im.convert("RGBA")
    data = numpy.asarray(im)
    data_flip = numpy.zeros(data.shape, dtype = data.dtype)
    data_flip[:,:,0] = data[:,:,2]
    data_flip[:,:,1] = data[:,:,1]
    data_flip[:,:,2] = data[:,:,0]
    data_flip[:,:,3] = data[:,:,3]
    im = Image.fromarray(data_flip.transpose())
    w,h = im.size[0], im.size[1]
    out += "data['" + i + "'] = " + repr(im.tobytes()) + " \n"
    out += "width['" + i + "'] = " + str(w) + "\n"
    out += "height['" + i + "'] = " + str(h) + "\n"

open(sys.argv[1], "w").write(out)
