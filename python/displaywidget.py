from PySide import QtGui, QtOpenGL, QtCore
from OpenGL import GL
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GL.ARB import texture_rg
from OpenGL.GL.ARB import half_float_vertex
from ctypes import c_void_p
import numpy
import math

vert_src = """#version 120
attribute vec2 positionIn;
attribute vec2 texIn;
varying vec2 texcoord;
void main()
{
    gl_Position= vec4(positionIn * 2.0 - vec2(1),0,1);
    texcoord = texIn;
}
"""

frag_src = """#version 120
uniform sampler2D texture;
uniform int hdr_mode;
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
    vec3 tex = texture2D(texture, coords).xyz;
    if (hdr_mode == 1) {
        gl_FragColor = vec4(convert(tex.x), convert(tex.y), convert(tex.z), 1);
    } else {
        gl_FragColor = vec4(tex,1);
    }
}
"""

class DisplayWidget(QtGui.QWidget):
    def __init__(self, parent):
        super(DisplayWidget, self).__init__(parent)
        
        self.buffers = None
        self.glWidget = GLWidget(self)       
              
        self.bufferComboBox = QtGui.QComboBox(self)
        policy = QtGui.QSizePolicy()
        policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self.bufferComboBox.setSizePolicy(policy)
        label = QtGui.QLabel("Buffers:")
        label.setBuddy(self.bufferComboBox)
        self.bufferComboBox.currentIndexChanged["QString"].connect(
            self.onBufferSelectChange)
        self.interactiveCheckBox = QtGui.QCheckBox("Interactive", self)
        midLayout = QtGui.QHBoxLayout()
        midLayout.addWidget(label)
        midLayout.addWidget(self.bufferComboBox)
        midLayout.addWidget(self.interactiveCheckBox)
                    
        self.label = QtGui.QLabel("Exposure: 0", self)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(-100,100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.onExposureChange)
        bottomLayout = QtGui.QHBoxLayout()
        bottomLayout.addWidget(self.label)
        bottomLayout.addWidget(self.slider)
        bottomLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.glWidget)
        layout.addLayout(midLayout)
        layout.addLayout(bottomLayout)
        self.setLayout(layout)

    def setBuffers(self, buffers):
        for i in xrange(self.bufferComboBox.count()):
            self.bufferComboBox.removeItem(0)
        self.buffers = buffers
        buffersList = buffers.keys()
        buffersList.sort()
        self.bufferComboBox.addItems(buffersList)

    def setActiveBuffer(self, bufferName):
        idx = self.bufferComboBox.findText(bufferName)
        if idx == self.bufferComboBox.currentIndex():
            self.refreshDisplay()
        else:
            self.bufferComboBox.setCurrentIndex(idx)
   
    def onBufferSelectChange(self, value):
        if str(value) in self.buffers:
            ndarray = self.buffers[str(value)].data
            self.glWidget.copyDataToTexture(ndarray)
            if ndarray.dtype == numpy.float32 or ndarray.dtype == numpy.float16:
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
        self.onBufferSelectChange(self.bufferComboBox.currentText())

    def sizeHint(self):
        return QtCore.QSize(400,400)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)
        
        self.w = 440
        self.h = 440

        self.rightBtnDown = False

        self.texture = None
        self.texturedata = None
        self.shader = None
        self.hdr_mode = 0
        self.vbo = None
        
        self.scale = 0.5
        self.steps = 0
        self.cx = 0.5
        self.cy = 0.5
        
        self.gamma = 1.0/2.2
        self.exposure = 0

        self.zoomFactor = 1.35
        self.panFactor = 0.002

    def initializeGL(self):
        pass

    def copyDataToTexture(self, ndarray):
        # Update dimensions of widget
        self.texturedata = ndarray
        self.w = ndarray.shape[0]
        self.h = ndarray.shape[1]
        self.updateGeometry()

        # Generate new texture
        if not self.texture:
            try:
                self.texture = GL.glGenTextures(1)
            except Exception:
                return
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
            self.hdr_mode = 1

            # The internal format changes with floating point textures
            if channels == 1:
                glInternalFormat = texture_rg.GL_R32F
            elif channels == 2:
                glInternalFormat = texture_rg.GL_RG32F
            elif channels == 3:
                glInternalFormat = GL.GL_RGB32F
            elif channels == 4:
                glInternalFormat = GL.GL_RGBA32F
        elif ndarray.dtype == numpy.float16:
            glType = GL.GL_FLOAT
            # Need to use the exposure shader if floating point
            self.hdr_mode = 1

            # The internal format changes with floating point textures
            if channels == 1:
                glInternalFormat = texture_rg.GL_R16F
            elif channels == 2:
                glInternalFormat = texture_rg.GL_RG16F
            elif channels == 3:
                glInternalFormat = GL.GL_RGB16F
            elif channels == 4:
                glInternalFormat = GL.GL_RGBA16F
        else:
            glType = GL.GL_UNSIGNED_BYTE
            self.hdr_mode = 0
        
        # Copy data to texture
        GL.glTexImage2D(target, 0, glInternalFormat, self.w, self.h,
                        0, glFormat, glType, ndarray)
        GL.glBindTexture(target, 0)
          
    def resizeGL(self, width, height):
        GL.glViewport(0,0,width,height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0,1,0,1,0,1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
      
    def compileShaders(self):
         # Build shaders
        vert_shader = shaders.compileShader(vert_src, GL.GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(frag_src, GL.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vert_shader, frag_shader)
        
    def paintGL(self):
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == 33305:
            return
        
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        if not self.texture:
            return
            
        if not self.shader:
            self.compileShaders()

        if not self.vbo:
            self.vbo = GL.glGenBuffers(1)

        shaders.glUseProgram(self.shader)
               
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        vertices = numpy.array(
            [-self.scale + self.cx, -self.scale + self.cy ,
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
              
        def _uniformLoc(name):
            return GL.glGetUniformLocation(self.shader,name)
        GL.glUniform1f(_uniformLoc("g"), self.gamma);
        GL.glUniform1f(_uniformLoc("m"), math.pow(2, self.exposure + 2.47393))
        GL.glUniform1f(_uniformLoc("s"), math.pow(2, -3.5 * self.gamma))
        GL.glUniform1i(_uniformLoc("hdr_mode"), self.hdr_mode);
        GL.glUniform1i(_uniformLoc("texture"), 0);
        GL.glActiveTexture(GL.GL_TEXTURE0);
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        GL.glDrawArrays(GL.GL_QUADS, 0, 4);

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        loc = GL.glGetAttribLocation(self.shader, "positionIn")
        GL.glDisableVertexAttribArray(loc)
        loc = GL.glGetAttribLocation(self.shader, "texIn")
        GL.glDisableVertexAttribArray(loc)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        shaders.glUseProgram(0)
        
        if self.rightBtnDown:
            self.renderPixelInfo()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        
        if event.button()== QtCore.Qt.RightButton:
            self.rightBtnDown = True
            self.glDraw()
            
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.rightBtnDown = False
            self.glDraw()
                
    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.cx += self.panFactor*dx
            self.cy -= self.panFactor*dy
            self.correctCenterCoordinates()
        
        self.lastPos = event.pos()
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

    def renderPixelInfo(self):
        # Get pixel positions px and py
        size = 2.0*(self.scale)
        offx = self.w * (self.scale - self.cx) / size
        offy = self.h * (self.scale - self.cy) / size
        px = int(offx + (self.lastPos.x() * self.w) / (self.width() * size))
        py = int(offy + (self.lastPos.y() * self.h) / (self.height()* size))
        py = self.h - py
        px = min(max(px,0), self.w - 1)
        py = min(max(py,0), self.h - 1)
        
        val = [None, None, None, None]
        for i in xrange(self.texturedata.shape[2]):
            val[i] = self.texturedata[px][py][i]
        texts = ["x:%i y:%i" % (px,py), 
                 "R:%f" % val[0] if val[0] else "n/a", 
                 "G:%f" % val[1] if val[1] else "n/a",
                 "B:%f" % val[2] if val[2] else "n/a"]
        font = QtGui.QFont()
        font.setFamily("Monospace")
        #font.setFixedPitch(True);
        metrics = QtGui.QFontMetrics(font)
        sx = 20 # spacing variable
        w,h = metrics.width(texts[1]), metrics.height()
        metrics.width(" ")
        x,y  = self.lastPos.x(), self.height() - self.lastPos.y() - sx
        dx,dy = 1.0/self.width(), 1.0/self.height()
            
        # Calculate pixel info position
        # Swap position if outside screen
        if x + 1.5*sx + w < self.width():
            x0 = x + 0.75*sx
            x1 = x + 1.5*sx + w + 10
            tx = x + sx
        else:
            x0 = x - 0.75*sx
            x1 = x - 1.5*sx - w
            tx = x - sx - w
        if y + sx - 5 * h > 0:
            y0 = y + sx
            y1 = y + sx - 5 * h
            ty = self.height()-y
        else:
            y0 = y - sx + 3 * h
            y1 = y - sx + 8 * h
            ty = self.height()-y - 5 * h - 0.5*sx

        # Draw transparent quad
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        GL.glEnable(GL.GL_BLEND)
        GL.glBegin(GL.GL_QUADS)
        GL.glColor4f(0,0,0,0.8)
        for x,y in zip([x0,x1,x1,x0],[y0,y0,y1,y1]):
            GL.glVertex2f(x * dx, y * dy)
        GL.glEnd()
        GL.glDisable(GL.GL_BLEND)
        
        # Render text
        GL.glColor4f(1,1,1,1)
        for i,text in enumerate(texts):
            self.renderText(tx, ty + i*h, text, font)
              

    
