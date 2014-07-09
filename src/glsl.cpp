#include "glsl.h"
#include "glsl_error.h"
#include <string.h>
#include <memory>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline GLenum _GetType(const Buffer & b);
//----------------------------------------------------------------------------//
inline GLenum _GetFormat(const Buffer & b);
//----------------------------------------------------------------------------//
inline GLenum _GetInternalFormat(const Buffer & b);
//----------------------------------------------------------------------------//
Base * CreateGLSL(unsigned int width, unsigned int height)
{
    return new GLSLImpl(width, height);
}
//----------------------------------------------------------------------------//
GLSLImpl::GLSLImpl(unsigned int width, unsigned int height)
        : Base(gpuip::GLSL, width, height)
{
    int argc = 1;
    std::auto_ptr<char> argv(new char[1]);
    char * argvp = argv.get();
    glutInit(&argc, &argvp);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(1,1);
    glutCreateWindow("");
    glewInit();

    if (glGetError() != GL_NO_ERROR) {
        throw std::logic_error("gpuip::GLSLImpl() error in init.");
    }
}
//----------------------------------------------------------------------------//
bool GLSLImpl::InitBuffers(std::string * err)
{
    std::map<std::string,Buffer>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);

        // No linear interpolation
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // Out of bounds -> Value at the border
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Mipmaps not needed.
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE); 

        // Allocate memory on gpu (needed to bind framebuffer)
        const Buffer & b = it->second;
        glTexImage2D(GL_TEXTURE_2D, 0, _GetInternalFormat(b),
                     _w, _h, 0, _GetFormat(b), _GetType(b), 0);
        
        _textures[it->second.name] = texID;

        if (_glErrorCreateTexture(err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool GLSLImpl::Build(std::string * err)
{
    // Simple vert shader code for a quad with texture coordinates
    static const char * vert_shader_code = 
            "attribute vec2 positionIn;\n"
            "varying vec2 texcoord;\n"
            "void main()\n"
            "{\n"
            "      gl_Position = vec4(vec2(-1) + 2*positionIn,0, 1);\n"
            "      texcoord = positionIn;\n"
            "}";

    GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    int length = strlen(vert_shader_code);
    glShaderSource(vertexShaderID, 1, &vert_shader_code, &length);
    glCompileShader(vertexShaderID);

    _shaders.resize(_kernels.size());
    for (int i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i]->code.c_str();
        const char * name = _kernels[i]->name.c_str();

        GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        length = strlen(code);
        glShaderSource(fragShaderID, 1, &code, &length);
        glCompileShader(fragShaderID);
  
        _shaders[i] = glCreateProgram();
        glAttachShader(_shaders[i], fragShaderID);
        glAttachShader(_shaders[i], vertexShaderID);
        glLinkProgram(_shaders[i]);

        if(_glCheckBuildError(_shaders[i], fragShaderID, err)) {
            return false;
        }
    }
           
    // Create FBOs
    _fbos.reserve(_kernels.size());
    glGenFramebuffers(_kernels.size(), _fbos.data());

    // Create a renderbuffer object to store depth info
    GLuint rboId;
    glGenRenderbuffers(1, &rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, rboId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _w, _h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Attach the textures to FBOs color attachment points
    for (int i = 0; i < _kernels.size(); ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbos[i]);
        for (int j = 0; j < _kernels[i]->outBuffers.size(); ++j) {
            const GLuint texID = _textures[_kernels[i]->outBuffers[j]];
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + j, 
                                   GL_TEXTURE_2D, texID, 0 /*mipmap level*/);
        }
       
        // attach the renderbuffer to depth attachment point
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, rboId);
        if (_glErrorFramebuffer(err)) {
            return false;
        }
    }
    
    // Create and build quad vbo
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    float vertices[] = {0,0,1,0,1,1,0,1};
    glBufferData(GL_ARRAY_BUFFER, 32, vertices, GL_STATIC_DRAW);

    return true;
}
//----------------------------------------------------------------------------//
bool GLSLImpl::Process(std::string * err)
{
    // Get old (current one) viewport coordinates and values
    GLint oldVp[4];
    glGetIntegerv(GL_VIEWPORT, oldVp);

    // Set the viewport to match the width and height
    glViewport(0, 0, _w, _h);

    for (int i = 0; i < _kernels.size(); ++i) {
        if (!_DrawQuad(*_kernels[i].get(), _fbos[i], _shaders[i], err)) {
            return false;
        }
    }

    // Reset back to the previous viewport
    glViewport(oldVp[0], oldVp[1], oldVp[2], oldVp[3]);
    return true;
}
//----------------------------------------------------------------------------//
bool GLSLImpl::Copy(const std::string & buffer,
                    Buffer::CopyOperation op,
                    void * data,
                    std::string * err)
{
    const Buffer & b = _buffers[buffer];
    if (op == Buffer::READ_DATA) {
        glBindTexture(GL_TEXTURE_2D, _textures[buffer]);
        glGetTexImage(GL_TEXTURE_2D, 0, _GetFormat(b), _GetType(b), data);
    } else if (op == Buffer::WRITE_DATA) {
        glBindTexture(GL_TEXTURE_2D, _textures[buffer]);
        glTexImage2D(GL_TEXTURE_2D, 0, _GetInternalFormat(b),
                     _w, _h, 0, _GetFormat(b), _GetType(b), data);
    }
    if (_glErrorCopy(err, buffer, op)) {
        return false;
    }
    return true;
}
//----------------------------------------------------------------------------//
bool GLSLImpl::_DrawQuad(const Kernel & kernel,
                   GLuint fbo,
                   GLuint shader,
                   std::string * error)
{   
    // Bind framebuffer and clear previous content
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Tell OpenGL how many buffers to draw
    std::vector<GLenum> enums;
    for (int i = 0; i < kernel.outBuffers.size(); ++i) {
        enums.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(enums.size(), &enums[0]);

    // Load shader (kernel)
    glUseProgram(shader);

    // Bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    
    // Coordinates for drawing the quad
    GLint loc = glGetAttribLocation(shader, "positionIn");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, 0, 8, 0);
       
    // Uniform data setup
    for (size_t i = 0; i < kernel.paramsInt.size(); ++i) {
        loc = glGetUniformLocation(shader, kernel.paramsInt[i].name.c_str());
        glUniform1i(loc, kernel.paramsInt[i].value);
    }
    for (size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
        loc = glGetUniformLocation(shader, kernel.paramsFloat[i].name.c_str());
        glUniform1f(loc, kernel.paramsFloat[i].value);
    }

    // Save current active texture 
    GLint activeTexture;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);

    // Texture setup
    for (int i = 0; i < kernel.inBuffers.size(); ++i) {
        loc = glGetUniformLocation(shader, kernel.inBuffers[i].c_str());
        glUniform1i(loc, i);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, _textures[kernel.inBuffers[i]]);
    }

    if (_glErrorDrawSetup(error, kernel.name)) {
        return false;
    }

    // Draw quad
    glDrawArrays(GL_QUADS, 0, 4);

    if (_glErrorDraw(error, kernel.name)) {
        return false;
    }
    
    // Unload framebuffer, vbo and shader
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0); 
    
    // Back to old active texture
    glActiveTexture(activeTexture);

    return true;
}
//----------------------------------------------------------------------------//
GLenum _GetType(const Buffer & b)
{
    switch(b.bpp / b.channels) {
        case 1:
            return GL_UNSIGNED_BYTE;
        case 2:
            return GL_HALF_FLOAT;
        case 4:
            return GL_FLOAT;
        case 8:
            return GL_DOUBLE;
        default:
            std::cerr << "Unknown OpenGL data type.. " << std::endl;
            return GL_FLOAT;
    }
}
//----------------------------------------------------------------------------//
GLenum _GetFormat(const Buffer & b)
{
    switch(b.channels) {
        case 1:
            return GL_RED;
        case 2:
            return GL_RG;
        case 3:
            return GL_RGB;
        case 4:
            return GL_RGBA;
        default:
            std::cerr << "Unknown OpenGL buffer channels " << b.channels
                      << std::endl;
            return GL_RGB;
    }
}
//----------------------------------------------------------------------------//
GLenum _GetInternalFormat(const Buffer & b)
{
    const GLenum type = _GetType(b);
    const GLenum format = _GetFormat(b);
    if (type == GL_HALF_FLOAT) {
        switch(format){
            case GL_RED:
                return GL_R16F;
            case GL_RG:
                return GL_RG16F;
            case GL_RGB:
                return GL_RGB16F;
            case GL_RGBA:
                return GL_RGBA16F;
            default:
                break;
        }
    } else if (type == GL_FLOAT) {
        switch(format){
            case GL_RED:
                return GL_R32F;
            case GL_RG:
                return GL_RG32F;
            case GL_RGB:
                return GL_RGB32F;
            case GL_RGBA:
                return GL_RGBA32F;
            default:
                break;
        }
    }
    return format;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
