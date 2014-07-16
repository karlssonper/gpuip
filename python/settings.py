from xml.dom import minidom
import pyGpuip as gpuip

class Settings(object):
    class Buffer(object):
        def __init__(self, name, type, channels):
            self.name = name
            self.type = type
            self.channels = channels
            self.input = ""
            self.output = ""

    class Param(object):
        def __init__(self, name, type, default, min, max):
            self.name = name
            self.type = type
            self.default = default if type == "float" else int(default)
            self.min = min if type == "float" else int(min)
            self.max = max if type == "float" else int(max)
            self.value = self.default if type == "float" else int(default)

    class Kernel(object):
        class KernelBuffer(object):
            def __init__(self, name, buffer):
                self.name = name
                self.buffer = buffer
        def __init__(self, name, code_file):
            self.name = name
            self.code = ""
            self.code_file = code_file
            self.params = []
            self.inBuffers = []
            self.outBuffers = []

    def __init__(self):
        self.buffers = []
        self.kernels = []
        self.environment = ""

    def read(self, xml_file):
        xmldom = minidom.parse(xml_file)
        
        # Environment
        self.environment = str(self.data(
                xmldom.getElementsByTagName("gpuip")[0],
                "environment"))
        
        # Buffers
        for b in xmldom.getElementsByTagName("buffer"):
            buffer = Settings.Buffer(self.data(b, "name"),
                                     self.data(b, "type"),
                                     eval(self.data(b, "channels")))
            if b.getElementsByTagName("input"):
                buffer.input = self.data(b, "input")
            if b.getElementsByTagName("output"):
                buffer.output = self.data(b, "output")
            self.buffers.append(buffer)

        # Kernels
        for k in xmldom.getElementsByTagName("kernel"):
            kernel = Settings.Kernel(self.data(k, "name"),
                                     self.data(k, "code_file"))
            
            kernel.code = open(kernel.code_file, "r").read()

            # In Buffers
            for inb in k.getElementsByTagName("inbuffer"):
                name = self.data(inb, "name")
                hasBuffer = inb.getElementsByTagName("targetbuffer")
                buf = self.data(inb, "targetbuffer") if hasBuffer else ""
                kernel.inBuffers.append(Settings.Kernel.KernelBuffer(name,buf))
                
            # Out Buffers
            for outb in k.getElementsByTagName("outbuffer"):
                name = self.data(outb, "name")
                hasBuffer = outb.getElementsByTagName("targetbuffer")
                buf = self.data(outb, "targetbuffer") if hasBuffer else ""
                kernel.outBuffers.append(Settings.Kernel.KernelBuffer(name,buf))

            # Params
            for p in k.getElementsByTagName("param"):
                param = Settings.Param(self.data(p, "name"),
                                       self.data(p, "type"),
                                       eval(self.data(p, "default")),
                                       eval(self.data(p, "min")),
                                       eval(self.data(p, "max")))
                param.value = eval(self.data(p, "value"))
                kernel.params.append(param)
            self.kernels.append(kernel)

    def write(self, xml_file):
        doc = minidom.Document()
        root = doc.createElement("gpuip")

        # Environment
        node = doc.createElement("environment")
        root.appendChild(node)
        node.appendChild(doc.createTextNode(self.environment))
        
        # Buffers
        bufferAttrs = ["name", "type", "channels", "input", "output"]
        for b in self.buffers:
            bufferNode = doc.createElement("buffer")
            root.appendChild(bufferNode)
            
            for attr in bufferAttrs:
                value = str(getattr(b, attr))
                if value != "":
                    node = doc.createElement(attr)
                    bufferNode.appendChild(node)
                    node.appendChild(doc.createTextNode(value))

        # Kernels
        paramAttrs = ["name", "type", "value", "default", "min", "max"]
        for k in self.kernels:
             # Write kernel code to file
            open(k.code_file, "w").write(k.code.replace("\t","    "))

            kernelNode = doc.createElement("kernel")
            root.appendChild(kernelNode)
            
            node = doc.createElement("name")
            kernelNode.appendChild(node)
            node.appendChild(doc.createTextNode(k.name))

            node = doc.createElement("code_file")
            kernelNode.appendChild(node)
            node.appendChild(doc.createTextNode(k.code_file))

            # In Buffers
            for inb in k.inBuffers:
                inbufferNode = doc.createElement("inbuffer")
                kernelNode.appendChild(inbufferNode)
                node = doc.createElement("name")
                inbufferNode.appendChild(node)
                node.appendChild(doc.createTextNode(inb.name))
                if inb.buffer != "":
                   node = doc.createElement("targetbuffer")
                   inbufferNode.appendChild(node)
                   node.appendChild(doc.createTextNode(inb.buffer)) 

            # In Buffers
            for outb in k.outBuffers:
                outbufferNode = doc.createElement("outbuffer")
                kernelNode.appendChild(outbufferNode)
                node = doc.createElement("name")
                outbufferNode.appendChild(node)
                node.appendChild(doc.createTextNode(outb.name))
                if outb.buffer != "":
                   node = doc.createElement("targetbuffer")
                   outbufferNode.appendChild(node)
                   node.appendChild(doc.createTextNode(outb.buffer)) 

            # Params
            for p in k.params:
                paramNode = doc.createElement("param")
                kernelNode.appendChild(paramNode)
                for attr in paramAttrs:
                    node = doc.createElement(attr)
                    paramNode.appendChild(node)
                    node.appendChild(doc.createTextNode(str(getattr(p, attr))))

        # Ugly result :(
        #root.writexml(open(xml_file,'w'), addindent="  ", newl='\n')

        # Work-around to get one line text nodes, taken from 
        #http://stackoverflow.com/questions/749796/pretty-printing-xml-in-python
        import re
        xml = root.toprettyxml(indent="  ")
        text_re = re.compile('>\n\s+([^<>\s].*?)\n\s+</', re.DOTALL)    
        file_handle = open(xml_file, 'w')
        file_handle.write(text_re.sub('>\g<1></', xml))
        file_handle.close()

    @staticmethod
    def data(n, name):
        return str(n.getElementsByTagName(name)[0].childNodes[0].data).strip()

    def create(self):
        if self.environment == "OpenCL":
            env = gpuip.Environment.OpenCL
        elif self.environment == "CUDA":
            env = gpuip.Environment.CUDA
        elif self.environment == "GLSL":
            env = gpuip.Environment.GLSL
        gpuip_obj = gpuip.gpuip(env)

        # Create and add buffers
        buffers = {}
        for b in self.buffers:
            buf = gpuip.Buffer()
            buf.name = b.name
            buf.channels = b.channels

            if b.type == "float":
                buf.bpp = 4 * b.channels
            elif b.type == "uchar":
                buf.bpp = 1 * b.channels

            buffers[b.name] = buf
            gpuip_obj.AddBuffer(buf)

        # Create kernels
        kernels = []
        for k in self.kernels:
            kernel = gpuip_obj.CreateKernel(k.name)
            kernels.append(kernel)
  
        # Set buffer linking and parameters for each kernels
        self.updateKernels(gpuip, buffers, kernels)

        return gpuip_obj, buffers, kernels
            
    def updateKernels(self, gpuip, buffers, kernels):
        for kernel, k in zip(kernels, self.kernels):
            # If no buffers were added but kernels have buffers -> error
            if not len(buffers) and (len(k.inBuffers) or len(k.outBuffers)):
                raise Exception("no buffers found")

            # Backup buffer if no buffer is set in kernels
            firstBuf = buffers.values()[0] if len(buffers) else None
            
            # Input buffers
            for inb in k.inBuffers:
                buf = buffers[inb.buffer] if inb.buffer != "" else firstBuf
                kernel.SetInBuffer(inb.name, buf)

            # Output buffers
            for outb in k.outBuffers:
                buf = buffers[outb.buffer] if outb.buffer != "" else firstBuf
                kernel.SetOutBuffer(outb.name, buf)

            # Params
            for p in k.params:
                if p.type == "float":
                    param = gpuip.ParamFloat()
                elif p.type == "int":
                    param = gpuip.ParamInt()
                param.name = p.name
                param.value = p.value
                kernel.SetParam(param)

            # Code
            kernel.code = k.code
