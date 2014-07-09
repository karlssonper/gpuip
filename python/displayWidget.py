from PySide import QtGui, QtOpenGL, QtCore
from OpenGL import GL
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
import OpenEXR
import Imath
import sys
import numpy
import math

vert_src = """
attribute vec2 positionIn;
varying vec2 texcoord;
void main()
{
      gl_Position=vec4(vec2(-1) + 2*positionIn,0, 1);
      texcoord = positionIn;
};
"""

frag_src = """
uniform sampler2D hdr_texture;
uniform float g;
uniform float m;
uniform float s;
varying vec2 texcoord;

float convert(float x)
{
    return clamp(pow(x*m,g) *s, 0.0, 1.0);
}

void main()
{
    vec2 coords = vec2(texcoord.x, 1.0 - texcoord.y);
    vec3 hdr = texture2D(hdr_texture, coords).xyz;
    gl_FragColor = vec4(convert(hdr.x), convert(hdr.y), convert(hdr.z), 1);
}
"""

#move to utils
def exrToNumpy(filename):
    # Open EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get width and height 
    dw = exr_file.header()['dataWindow']
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # Read data from file
    imath_float = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [exr_file.channel(c, imath_float) for c in "RGB"]

    # Convert to 1D numpy arrays
    rgb = [numpy.fromstring(c,dtype=numpy.float32) for c in channels]

    # Create 3D numpy array
    npyArray = numpy.zeros((width,height,3), dtype = numpy.float32, order = "C")

    # Copy from 1D arrays to the 3D
    for i in range(3):
        npyArray [:,:,i] = rgb[i].reshape(width,height)

    return npyArray

class DisplayWidget(QtGui.QWidget):
    def __init__(self, parent):
        super(DisplayWidget, self).__init__(parent)

        layout = QtGui.QVBoxLayout()

        topWidget = QtGui.QWidget(self)
        topLayout = QtGui.QHBoxLayout()
        label = QtGui.QLabel("Buffers:")
        bufferOptions = QtGui.QComboBox(topWidget)
        bufferOptions.addItems(["lol\t", "hey\t"])
        topLayout.addWidget(label)
        topLayout.addWidget(bufferOptions)
        topWidget.setLayout(topLayout)
        
        self.glWidget = GLWidget(0,0, self)        
                
        bottomWidget = QtGui.QWidget(self)
        bottomLayout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel("Exposure: 0")
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(-100,100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.onExposureChange)
        bottomLayout.addWidget(self.label)
        bottomLayout.addWidget(self.slider)
        bottomLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        bottomWidget.setLayout(bottomLayout)

        topWidget.setSizePolicy(QtGui.QSizePolicy.Minimum,
                                QtGui.QSizePolicy.Preferred)
        self.glWidget.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Preferred)
        bottomWidget.setSizePolicy(QtGui.QSizePolicy.Minimum,
                                QtGui.QSizePolicy.Preferred)

        layout.addWidget(topWidget)
        layout.addWidget(self.glWidget)
        layout.addWidget(bottomWidget)

        self.setLayout(layout)

    def onExposureChange(self):
        value = 0.1 * self.slider.value()
        self.glWidget.exposure = value
        self.label.setText("Exposure: " + str(value))        
        self.glWidget.glDraw()

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, width, height, parent):
        super(GLWidget, self).__init__(parent)
        self.width = width
        self.height = height

        self.texture = None
        self.shader = None
        self.vbo = None

        self.gamma = 1.0/2.2
        self.exposure = 0

    def initializeGL(self):
        # Create texture from HDR
        self.texture = GL.glGenTextures(1)
        target = GL.GL_TEXTURE_2D
        GL.glBindTexture(target, self.texture)
        GL.glTexParameterf(target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameterf(target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameterf(target, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameterf(target, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(target, GL.GL_GENERATE_MIPMAP, GL.GL_FALSE);
        data = exrToNumpy("/home/per/dev/exr_test/GoldenGate.exr")
        self.width = data.shape[0]
        self.height = data.shape[1]
        self.updateGeometry()
        GL.glTexImage2D(target, 0, GL.GL_RGB32F, data.shape[0], data.shape[1],
                        0, GL.GL_RGB, GL.GL_FLOAT, data)

        # Build shaders
        vert_shader = shaders.compileShader(vert_src, GL.GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(frag_src, GL.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vert_shader, frag_shader)
        
        # Build quad vbo
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo);
        vertices = numpy.array([0,0,1,0,1,1,0,1], dtype = numpy.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 32, vertices, GL.GL_STATIC_DRAW);

    def resizeGL(self, width, height):
        GL.glViewport(0,0,width,height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0,1,0,1,0,1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        shaders.glUseProgram(self.shader)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo);
        loc = GL.glGetAttribLocation(self.shader, "positionIn")
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, 0, 8, None)

        loc = GL.glGetUniformLocation(self.shader, "hdr_texture")
        GL.glUniform1i(loc, 0);
        GL.glActiveTexture(GL.GL_TEXTURE0);
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        loc = GL.glGetUniformLocation(self.shader, "g")
        GL.glUniform1f(loc, self.gamma);
        loc = GL.glGetUniformLocation(self.shader, "m")
        GL.glUniform1f(loc, math.pow(2, self.exposure + 2.47393))
        loc = GL.glGetUniformLocation(self.shader, "s")
        GL.glUniform1f(loc, math.pow(2, -3.5 * self.gamma))
        GL.glDrawArrays(GL.GL_QUADS, 0, 4);

    def sizeHint(self):
        return QtCore.QSize(self.width,self.height)
