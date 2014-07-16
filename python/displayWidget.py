from PySide import QtGui, QtOpenGL, QtCore
from OpenGL import GL
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from ctypes import c_void_p
import OpenEXR
import Imath
import sys
import numpy
import math

vert_src = """
attribute vec2 positionIn;
attribute vec2 texIn;
varying vec2 texcoord;
void main()
{
      gl_Position= vec4(2*positionIn - vec2(1),0,1);
      texcoord = texIn;
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

def pngToNumpy(filename):
    from PIL import Image
    image = Image.open(filename)
    return numpy.array(image).transpose((1,0,2)).copy(order="C")

class DisplayWidget(QtGui.QWidget):
    def __init__(self, parent):
        super(DisplayWidget, self).__init__(parent)
        layout = QtGui.QVBoxLayout()

        self.bufferData = {}
        self.glWidget = GLWidget(self)       
        self.glWidget.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Preferred)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum,
                           QtGui.QSizePolicy.Minimum)

        
        midLayout = QtGui.QHBoxLayout()
        self.bufferComboBox = QtGui.QComboBox(self)
        policy = QtGui.QSizePolicy()
        policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self.bufferComboBox.setSizePolicy(policy)
        label = QtGui.QLabel("Buffers:")
        label.setBuddy(self.bufferComboBox)
        self.bufferComboBox.activated["QString"].connect(
            self.onBufferSelectChange)
        midLayout.addWidget(label)
        midLayout.addWidget(self.bufferComboBox)
        #midLayout.addStretch()
                
        bottomLayout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel("Exposure: 0", self)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(-100,100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.onExposureChange)
        bottomLayout.addWidget(self.label)
        bottomLayout.addWidget(self.slider)
        bottomLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
   
        layout.addWidget(self.glWidget)
        layout.addLayout(midLayout)
        layout.addLayout(bottomLayout)

        self.setLayout(layout)

    def setBufferData(self, bufferData):
        self.bufferData = bufferData
        self.bufferComboBox.addItems(bufferData.keys())
    
    def setDisplayDebug(self, displayDebug):
        self.glWidget.setDisplayDebug(displayDebug)

    def onBufferSelectChange(self, value):
        ndarray = self.bufferData[str(value)]
        self.glWidget.copyDataToTexture(ndarray)
        if ndarray.dtype == numpy.float32:
            self.slider.setEnabled(True)
        else:
            self.slider.setEnabled(False)
        self.glWidget.glDraw()

    def onExposureChange(self):
        value = 0.1 * self.slider.value()
        self.glWidget.exposure = value
        self.label.setText("Exposure: " + str(value))        
        self.glWidget.glDraw()

    def refreshDisplay(self):
        bufferName = str(self.bufferComboBox.currentText())
        self.glWidget.copyDataToTexture(self.bufferData[bufferName])

    def sizeHint(self):
        return QtCore.QSize(400,400)



class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)
        self.w = 440
        self.h = 440

        self.displayDebug = None

        self.texture = None
        self.shader = None
        self.useShader = False
        self.vbo = None
        
        self.scale = 0.5
        self.steps = 0
        self.cx = 0.5
        self.cy = 0.5
        
        self.gamma = 1.0/2.2
        self.exposure = 0

        self.zoomFactor = 1.35
        self.panFactor = 0.002

    def setDisplayDebug(self, displayDebug):
        self.displayDebug = displayDebug

    def initializeGL(self):
        self.texture = GL.glGenTextures(1)

        # Build shaders
        vert_shader = shaders.compileShader(vert_src, GL.GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(frag_src, GL.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vert_shader, frag_shader)
        
        self.vbo = GL.glGenBuffers(1)
                
    def copyDataToTexture(self, ndarray):
        # Update dimensions of widget
        self.w = ndarray.shape[0]
        self.h = ndarray.shape[1]
        self.updateGeometry()

        # Generate new texture
        target = GL.GL_TEXTURE_2D
        GL.glBindTexture(target, self.texture)
        GL.glTexParameterf(target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameterf(target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameterf(target, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameterf(target, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(target, GL.GL_GENERATE_MIPMAP, GL.GL_FALSE);
        
        # Get texture format
        channels = ndarray.shape[2] if ndarray.ndim == 3 else 1
        if channels == 1:
            glFormat = GL.GL_RED
        elif channels == 2:
            glFormat = GL.GL_RG
        elif channels == 3:
            glFormat = GL.GL_RGB
        elif channels == 4:
            glFormat = GL.GL_RGBA
        glInternalFormat = glFormat

        # Get texture type
        if ndarray.dtype == numpy.float32:
            glType = GL.GL_FLOAT
            # Need to use the exposure shader if floating point
            self.useShader = True

            # The internal format changes with floating point textures
            if channels == 1:
                glInternalFormat = GL.GL_RED32F
            elif channels == 2:
                glInternalFormat = GL.GL_RG32F
            elif channels == 3:
                glInternalFormat = GL.GL_RGB32F
            elif channels == 4:
                glInternalFormat = GL.GL_RGBA32F
        else:
            glType = GL.GL_UNSIGNED_BYTE
            self.useShader = False
        
        # Copy data to texture
        GL.glTexImage2D(target, 0, glInternalFormat, self.w, self.h,
                        0, glFormat, glType, ndarray)
  
    def resizeGL(self, width, height):
        GL.glViewport(0,0,width,height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0,1,0,1,0,1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self.useShader:
            shaders.glUseProgram(self.shader)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            vertices = numpy.array(
                [-self.scale + self.cx, -self.scale + self.cy,
                  self.scale + self.cx, -self.scale + self.cy,
                  self.scale + self.cx, self.scale + self.cy,
                  -self.scale + self.cx, self.scale + self.cy,
                  0,0,1,0,1,1,0,1], dtype = numpy.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, 64, vertices, GL.GL_STATIC_DRAW)
            loc = GL.glGetAttribLocation(self.shader, "positionIn")
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, 0, 8, c_void_p(0))

            loc = GL.glGetAttribLocation(self.shader, "texIn")
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, 0, 8, c_void_p(32))
            
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

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            shaders.glUseProgram(0)
        else:
            # Draw a simple quad with texture coordinates
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glColor3f(1,1,1)
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(1,0); 
            GL.glVertex2f(-self.scale + self.cx,-self.scale + self.cy)
            GL.glTexCoord2f(1,1); 
            GL.glVertex2f(self.scale + self.cx,-self.scale + self.cy)
            GL.glTexCoord2f(0,1); 
            GL.glVertex2f(self.scale + self.cx, self.scale + self.cy)
            GL.glTexCoord2f(0,0);  
            GL.glVertex2f(-self.scale + self.cx, self.scale + self.cy)
            GL.glEnd()
            GL.glDisable(GL.GL_TEXTURE_2D)
          
    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.cx += self.panFactor*dx
            self.cy -= self.panFactor*dy
            self.correctCenterCoordinates()
        
        self.lastPos = event.pos()

        if self.displayDebug and event.buttons() & QtCore.Qt.RightButton:
            # Todo.
            ppx = event.pos().x() / float(self.width())
            ppy = event.pos().y() / float(self.height())

            sx = self.w / self.scale * 2 
            sy = self.h / self.scale * 2

            px = sx*(self.scale - self.cx + ppx)
            py = sy*(self.scale - self.cx + ppy)

            text = "Pixel coordinates: %f, %f" % (px, py)
            self.displayDebug.setPlainText(text)

        self.glDraw()

    def wheelEvent(self, event):
        if event.delta() > 0:
            self.steps += 1
        else:
            self.steps -= 1
            
        # Only allow inital zoom (not smaller)
        if self.steps < 0:
            self.steps = 0
        
        self.scale = 0.5 * math.pow(self.zoomFactor, self.steps)
        self.correctCenterCoordinates()
        self.glDraw()

    def correctCenterCoordinates(self):
        if -self.scale + self.cx > 0:
            self.cx = self.scale
        if self.scale + self.cx < 1:
            self.cx = 1 - self.scale
        if -self.scale + self.cy > 0:
            self.cy = self.scale
        if self.scale + self.cy < 1:
            self.cy = 1 - self.scale

    def sizeHint(self):
        return QtCore.QSize(self.w,self.h)
    
