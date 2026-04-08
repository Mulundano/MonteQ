# MonteQ readme

Necessary packages:
  * [qiskit['visualize']](https://docs.quantum.ibm.com/guides/install-qiskit)
  * rustworks
  * numpy
  * matplotlib( qiskit[visualize] comes with it)
    
**WARNING: current structure does not have error handling so inputs must be given in specified format**

## Instructions: 
Within the notebook labeled "src" in the example notebook, there as example of how to use the functions to produce a logical result and a hardware aware result.


**If you wish to test a full circuit output use the function "full_circuit" located in the src/mcts_module file and the pair_solve function in the src/heuristic_module file - "example" shows how to dynamically add the path into your system path so that src can be a module.**

The "full_circuit" function returns a full qiskit quantum circuit that implements a given simulation problem(presented as a list of Pauli words like ucc_2_4 for example). Its parameters are as follows:
  * p_word - list of Pauli stringss that will become a Pauli word
  * function - heuristic function that creates a path from a parent to a child (currently only the pair_solve heuristic can be used here)
  * stop_time - float to use with MCTS to dictate stop time in seconds
  * order_preserving - boolean value that dictates whether you preserve the order of Pauli strings up to commutation or not
  * sims - number of iterations
  * stop_time - total time to run - if sims is given it is prioritzed over stop-time
  * rot_params - list of floats which are the rotation parameters to use for each Pauli string - if None placeholders are made
   


This function returns a full qiskit [QuantumCircuit](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit) which can be displayed and a CNOT count can be acquired





