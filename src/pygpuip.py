import sys
import dl
sys.setdlopenflags(dl.RTLD_NOW | dl.RTLD_GLOBAL)
from _pygpuip import *
