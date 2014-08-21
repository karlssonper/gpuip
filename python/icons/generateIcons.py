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

out = """from PySide import QtGui
data, width, height = {}, {}, {}
def get(name):
    image = QtGui.QImage(data[name], width[name], height[name], 
                         QtGui.QImage.Format_ARGB32)
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))
"""

for i in glob.glob("*.png"):
    name = i[:i.find(".")]
    data = numpy.asarray(Image.open(i).convert("RGBA"))
    data_flip = numpy.array(data, copy=True)
    data_flip[:,:,0] = data[:,:,2]
    data_flip[:,:,2] = data[:,:,0]
    out += "data['" + name + "'] = " + repr(data_flip.tostring()) + " \n"
    out += "width['" + name + "'] = " + str(data_flip.shape[0]) + "\n"
    out += "height['" + name + "'] = " + str(data_flip.shape[1]) + "\n"

open(sys.argv[1], "w").write(out)
