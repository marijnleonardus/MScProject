from raytracing import *

path = ImagingPath()
path.append(Space(d=100))
path.append(Lens(f=50, diameter=25))
path.append(Space(d=120))
path.append(Lens(f=70))
path.append(Space(d=100))
path.display()