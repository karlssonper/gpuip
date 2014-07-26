from PIL import Image
import os
icons = {}
icons["pug"] = "resources/pug.png"
icons["new"] = "resources/new.png"
icons["newExisting"] = "resources/newExisting.png"
icons["open"] = "resources/open.png"
icons["save"] = "resources/save.png"
icons["refresh"] = "resources/refresh.png"
icons["build"] = "resources/build.png"
icons["import"] = "resources/import.png"
icons["process"] = "resources/process.png"
icons["export"] = "resources/export.png"
icons["init"] = "resources/init.png"

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
