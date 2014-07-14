from xml.dom import minidom

class Settings(object):
    class Buffer(object):
        def __init__(self, name, type, channels):
            self.name = str(name)
            self.type = str(type)
            self.channels = str(channels)

    class Param(object):
        def __init__(self, name, type, default, min, max):
            self.name = str(name)
            self.type = str(type)
            self.default = str(default)
            self.min = str(min)
            self.max = str(max)

    class Kernel(object):
        def __init__(self, name):
            self.name = str(name)
            self.params = []

    def __init__(self):
        self.buffers = []
        self.kernels = []

    def read(self, xml_file):
        xmldom = minidom.parse(xml_file)

        # Buffers
        for b in xmldom.getElementsByTagName("buffer"):
            buffer = Settings.Buffer(self.data(b, "name"),
                                     self.data(b, "type"),
                                     self.data(b, "channels"))
            self.buffers.append(buffer)

        # Kernels
        for k in xmldom.getElementsByTagName("kernel"):
            kernel = Settings.Kernel(self.data(k, "name"))
            
            # Params
            for p in xmldom.getElementsByTagName("param"):
                param = Settings.Param(self.data(p, "name"),
                                       self.data(p, "type"),
                                       self.data(p, "default"),
                                       self.data(p, "min"),
                                       self.data(p, "max"))
                kernel.params.append(param)
            self.kernels.append(kernel)

    def write(self, xml_file):
        doc = minidom.Document()
        root = doc.createElement("gpuip")

        bufferAttrs = ["name", "type", "channels"]
        for b in self.buffers:
            bufferNode = doc.createElement("buffer")
            root.appendChild(bufferNode)
            
            for attr in bufferAttrs:
                node = doc.createElement(attr)
                bufferNode.appendChild(node)
                textNode = doc.createTextNode(getattr(b, attr))
                node.appendChild(textNode)

        paramAttrs = ["name", "type", "default", "min", "max"]
        for k in self.kernels:
            kernelNode = doc.createElement("kernel")
            root.appendChild(kernelNode)
            node = doc.createElement("name")
            kernelNode.appendChild(node)
            textNode = doc.createTextNode(k.name)
            node.appendChild(textNode)

            for p in k.params:
                paramNode = doc.createElement("param")
                kernelNode.appendChild(paramNode)
                for attr in paramAttrs:
                    node = doc.createElement(attr)
                    paramNode.appendChild(node)
                    textNode = doc.createTextNode(getattr(p, attr))
                    node.appendChild(textNode)

        root.writexml(open(xml_file,'w'), addindent="  ", newl='\n')

    @staticmethod
    def data(element, name):
        return element.getElementsByTagName(name)[0].childNodes[0].data
            
