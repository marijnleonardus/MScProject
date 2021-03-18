# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:18:15 2021

@author: marijn
"""


from blueqat import Circuit

#CNOT/CX gate on |00> state
out1 = Circuit().cx[0,1].m[:].run(shots=1)
print(out1)

#CX gate on |10> by applying bitflip (X) gate on first qubit first
#yielding |10> after the first step
out2 = Circuit(2).x[0].cx[0,1].m[:].run(shots=1)
print(out2)