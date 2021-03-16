# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:14:39 2021

@author: marij
"""

import cirq

qubit = cirq.NamedQubit("mycubit")

circuit = cirq.Circuit(cirq.H(qubit))
print(circuit)

result = cirq.Simulator().simulate(circuit)
print("result:") 
print(result)