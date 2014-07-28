from PIL import Image
import os
icons = {}
icons["pug"] = "pug.png"
icons["new"] = "new.png"
icons["newExisting"] = "newExisting.png"
icons["open"] = "open.png"
icons["save"] = "save.png"
icons["refresh"] = "refresh.png"
icons["build"] = "build.png"
icons["import"] = "import.png"
icons["process"] = "process.png"
icons["export"] = "export.png"
icons["init"] = "init.png"

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
    os.system("convert " + icons[i] +" -channel rbgba -separate -swap 0,2 -combine tmp.png")

    im = Image.open("tmp.png")
    w,h = im.size[0], im.size[1]
    #out += "data['" + i + "'] = " + repr(im.tobytes('raw', 'RGBA')) + "\n"
    out += "data['" + i + "'] = " + repr(im.tobytes()) + " \n"
    out += "width['" + i + "'] = " + str(w) + "\n"
    out += "height['" + i + "'] = " + str(h) + "\n"
    os.remove("tmp.png")

f = open("icons.py", "w")
f.write(out)
