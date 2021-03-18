# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:59:40 2021

@author: marij
"""


from blueqat import Circuit

#Hamard gate/superposition,1000 tries

out00 = Circuit().h[0].m[:].run(shots = 1000)
print(out00)

#state vectors

#state vector on 1/sqrt(2)(|0>+|1>)
out_vector = Circuit().h[0].run()
print(out_vector)

#state vector on 1/sqrt(2)(|0>-|1>)
out_vector_z_first = Circuit().h[0].z[0].run()
print(out_vector_z_first)