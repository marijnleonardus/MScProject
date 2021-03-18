# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:41:38 2021

@author: marijn
"""

from blueqat import Circuit

#Toffoli gate on |00> control gates
out00 = Circuit().ccx[0,1,2].m[:].run(shots=1)
print(out00)

#Toffoli gate on |10> control gates
out10 = Circuit().x[0].ccx[0,1,2].m[:].run(shots=1)
print(out10)

#Toffoli gate on |11> control gates
out11 = Circuit().x[:2].ccx[0,1,2].m[:].run(shots=1)
print(out11)