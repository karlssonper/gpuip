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
Base * CreateGLSL()
{
    return new GLSLImpl();
}
//----------------------------------------------------------------------------//
GLSLImpl::GLSLImpl()
        : Base(gpuip::GLSL)
{
    if (!glfwInit()) {
         throw std::logic_error("gpuip could not initiate GLFW");
    }

    GLFWwindow *  window = glfwCreateWindow(1, 1, "", NULL, NULL);

    if (!window) {
        throw std::logic_error("gpuip could not create window with glfw");
    }

    glfwMakeContextCurrent(window);

    GLenum result = glewInit();
    if (result != GLEW_OK) {
        std::cerr << glewGetErrorString(result) << std::endl;
        throw std::logic_error("gpuip could not initiate GLEW");
    }
    
    if (glGetError() != GL_NO_ERROR) {
        throw std::logic_error("gpuip::GLSLImpl() error in init.");
    }

    // Simple vert shader code for a quad with texture coordinates
    static const char * vert_shader_code = 
            "#version 120\n"
            "attribute vec2 positionIn;\n"
            "varying vec2 x;\n"
            "void main()\n"
            "{\n"
            "      gl_Position = vec4(vec2(-1) + 2*positionIn,0, 1);\n"
            "      x = positionIn;\n"
            "}";

    _vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    int length = strlen(vert_shader_code);
    glShaderSource(_vertexShaderID, 1, &vert_shader_code, &length);
    glCompileShader(_vertexShaderID);
}
//----------------------------------------------------------------------------//
bool GLSLImpl::InitBuffers(std::string * err)
{
    std::map<std::string, GLuint>::iterator itt;
    if (!_fbos.empty()) {
        glDeleteRenderbuffers(1, &_rboId);
        glDeleteBuffers(_fbos.size(),_fbos.data());
    }
    for(itt = _textures.begin(); itt != _textures.end(); ++itt) {
        glDeleteTextures(1, &itt->second);
    }
    _fbos.clear();
    _textures.clear();
    
    
    std::map<std::string,Buffer>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
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
            
    // Create FBOs
    _fbos.reserve(_kernels.size());
    glGenFramebuffers(_kernels.size(), _fbos.data());

    // Create a renderbuffer object to store depth info
    GLuint _rboId;
    glGenRenderbuffers(1, &_rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, _rboId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _w, _h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Attach the textures to FBOs color attachment points
    for(size_t i = 0; i < _kernels.size(); ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbos[i]);
        for(size_t j = 0; j < _kernels[i]->outBuffers.size(); ++j) {
            GLuint texID = _textures[_kernels[i]->outBuffers[j].first.name];
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + j, 
                                   GL_TEXTURE_2D, texID, 0 /*mipmap level*/);
        }
       
        // attach the renderbuffer to depth attachment point
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, _rboId);
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
bool GLSLImpl::Build(std::string * err)
{
    for(size_t i = 0; i < _programs.size(); ++i) {
        glDeleteProgram(_programs[i]);
    }
    
    _programs.resize(_kernels.size());
    for(size_t i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i]->code.c_str();
        const GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        const int length = strlen(code);
        glShaderSource(fragShaderID, 1, &code, &length);
        glCompileShader(fragShaderID);
  
        _programs[i] = glCreateProgram();
        glAttachShader(_programs[i], fragShaderID);
        glAttachShader(_programs[i], _vertexShaderID);
        glLinkProgram(_programs[i]);
        glDeleteShader(fragShaderID);
        if(_glCheckBuildError(_programs[i], fragShaderID, err)) {
            return false;
        }
    }

    return true;
}
//----------------------------------------------------------------------------//
bool GLSLImpl::Process(std::string * err)
{
    glPushAttrib( GL_VIEWPORT_BIT );
    
    // Set the viewport to match the width and height
    glViewport(0, 0, _w, _h);

    for(size_t i = 0; i < _kernels.size(); ++i) {
        if (!_DrawQuad(*_kernels[i].get(), _fbos[i], _programs[i], err)) {
            return false;
        }
    }

    // Reset back to the previous viewport
    glPopAttrib();
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
std::string GLSLImpl::GetBoilerplateCode(Kernel::Ptr kernel) const
{
    std::stringstream ss;
    ss << "#version 120\n";
    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        ss << "uniform sampler2D " << kernel->inBuffers[i].second << ";\n";
    }
    for(size_t i = 0; i < kernel->paramsInt.size(); ++i) {
        ss << "uniform int " << kernel->paramsInt[i].name <<";\n";
    }
    for(size_t i = 0; i < kernel->paramsFloat.size(); ++i) {
        ss << "uniform float " << kernel->paramsFloat[i].name <<";\n";
    }
    ss << "varying vec2 x; // texture coordinates\n"
       << "uniform float dx; // delta\n"
       << "void main()\n"
       << "{\n";
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        if (i) {
            ss << "\n";
        }
        ss << "    // gl_FragData[" << i << "] is buffer "
           << kernel->outBuffers[i].second << "\n"
           << "    glFragData[" << i<<"] = vec4(0,0,0,1);\n";  
    }    
    ss << "}";
    return ss.str();
}
//----------------------------------------------------------------------------//
bool GLSLImpl::_DrawQuad(const Kernel & kernel,
                   GLuint fbo,
                   GLuint program,
                   std::string * error)
{   
    // Bind framebuffer and clear previous content
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Tell OpenGL how many buffers to draw
    std::vector<GLenum> enums;
    for(size_t i = 0; i < kernel.outBuffers.size(); ++i) {
        enums.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(enums.size(), &enums[0]);

    // Load program (kernel)
    glUseProgram(program);

    // Bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    
    // Coordinates for drawing the quad
    GLint loc = glGetAttribLocation(program, "positionIn");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, 0, 8, 0);
       
    // Uniform data setup
    for(size_t i = 0; i < kernel.paramsInt.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.paramsInt[i].name.c_str());
        glUniform1i(loc, kernel.paramsInt[i].value);
    }
    for(size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.paramsFloat[i].name.c_str());
        glUniform1f(loc, kernel.paramsFloat[i].value);
    }
    glUniform1f(glGetUniformLocation(program, "dx"), 1.0f/_w);

    // Save current active texture 
    GLint activeTexture;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);

    // Texture setup
    for(size_t i = 0; i < kernel.inBuffers.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.inBuffers[i].second.c_str());
        glUniform1i(loc, i);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, _textures[kernel.inBuffers[i].first.name]);
    }

    if (_glErrorDrawSetup(error, kernel.name)) {
        return false;
    }

    // Draw quad
    glDrawArrays(GL_QUADS, 0, 4);

    if (_glErrorDraw(error, kernel.name)) {
        return false;
    }
    
    // Unload framebuffer, vbo and program
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
