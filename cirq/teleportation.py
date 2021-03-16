# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:16:41 2021

@author: marijn
"""


# DEMO using the built in function for teleportation in cirq
# https://quantumai.google/cirq/tutorials/educators/textbook_algorithms

import cirq
import numpy as np

def make_quantum_teleportation_circuit(gate):
        #make empty circuit
        circuit = cirq.Circuit()
        
        # make 3 qubits
        msg = cirq.NamedQubit("Message")
        alice = cirq.NamedQubit("Alice")
        bob = cirq.NamedQubit("Bob")
        
        #prepare message to send
        circuit.append(gate(msg))
        
        #bell state share between alice and bob
        circuit.append([cirq.H(alice), cirq.CNOT(alice,bob)])
        
        #bell measurement of entangled qubit
        circuit.append([cirq.CNOT(msg, alice), cirq.H(msg), cirq.measure(msg, alice)])
        
        #recover bell measurement from 2 classical bits
        circuit.append([cirq.CNOT(alice, bob), cirq.CZ(msg, bob)])
        
        return circuit

#gate to put message qubit in some way to send    
gate = cirq.X ** 0.25

#create circuit
circuit = make_quantum_teleportation_circuit(gate)
print(circuit)

"""Checking by comparing initial qubit to Bob's qubit"""
message = cirq.Circuit(gate.on(cirq.NamedQubit("Message"))).final_state_vector()
message_bloch_vector = cirq.bloch_vector_from_state_vector(message, index = 0)
print("Bloch vector of message qubit:")
print(np.round(message_bloch_vector , 3))