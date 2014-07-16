import settings

s = settings.Settings()
s.environment = "OpenCL"

b1 = settings.Settings.Buffer("per", "float", 3)
b2 = settings.Settings.Buffer("color", "uchar", 1)
s.buffers = [b1,b2]

k1 = settings.Settings.Kernel("smooth", "smooth.cl")
k2 = settings.Settings.Kernel("gauss", "gauss.cl")

p1 = settings.Settings.Param("alpha", "float", 2, -10, 10)
p2 = settings.Settings.Param("beta", "float", 20, -100, 100)
p3 = settings.Settings.Param("c", "int", 20, -100, 100)

k1.params = [p1,p2,p3]
k2.params = [p1,p2,p3]

s.kernels = [k1,k2]

s.write("test.ip")
