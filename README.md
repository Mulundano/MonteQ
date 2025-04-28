# MCTS implementation readme

Necessary packages:
  * [qiskit['visualize']](https://docs.quantum.ibm.com/guides/install-qiskit)
  * networkX
  * numpy
  * matplotlib( qiskit[visualize] comes with it)
    
**WARNING: current structure does not have error handling so inputs must be given in specified format**

## Instructions: 
Within the notebook labeled "mcts_example" in the example folder, there are three lists labeled ucc_2_4, ucc_2_6, ucc_4_8. These are lists of Pauli words.
Each Pauli word is in the form of a list of Pauli strings (not in Tableau form). Within notebook, the time and UCT parameters are set early on. The following functions are best used in a jupyter notebook

**If you wish to test a full circuit output use the function "full_circuit" located in the src/mcts_module file and the pair_solve function in the src/heuristic_module file - "mcts_example" shows how to dynamically add the path into your system path so that src can be a module.**

The "full_circuit" function returns a full qiskit quantum circuit that implements a given simulation problem(presented as a list of Pauli words like ucc_2_4 for example). Its parameters are as follows:
  * p_sentence - list of Pauli words(each Pauli word is not in Tableau form)
  * function - heuristic function that creates a path from a parent to a child (currently only the pair_solve heuristic can be used here)
  * stop_time - float to use with MCTS to dictate stop time in seconds
  * rot_params - list of floats which are the rotation parameters to use for each Pauli string - if None placeholders are made
  * gate_cancellation: boolean value to dictate whether to apply [CommutativeCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.CommutativeCancellation) from Qiskit (removes gates that line up with their inverses directly - does not do any other optimizations)


This function returns a full qiskit [QuantumCircuit](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit) which can be displayed and a CNOT count can be acquired

**If you wish to test only for solutions of singular Pauli words use the function "p_word_solution" located in the src/mcts_module file.**

The "p_word_solution" function returns the "head" of a circuit in the form of a list of "action tuples":

  * head - within a quantum simulation circuit the beginning of a circuit contains clifford operation/s and the required Rz operation/s. The end of the circuit is the inverse of the clifford operations. The head is the clifford operation/s and the Rz operation/s. The idea is that the tail can be absorbed(see Clifford Absorption under [QuCLEAR](https://arxiv.org/pdf/2408.13316))
  * action tuple - this is a tuple in the form (operation, qubit/s operation is applied to)

The parameters are as follows:
  * p_word - Pauli word in Tableau form
  * function - heuristic function to use in MCTS
  * mcts_param - float to use with MCTS to use with UCT
  * end_time - float to use with MCTS to dictate stop time in seconds

**If you wish to test only test how the Monte Carlo Tree Search works use the "MCTS" function located in the src/mcts_module file.**

The "MCTS" function returns the best action to take (in the form of a list of "action tuples"), the child that is produced after that action is taken(in the form of a Pauli word) and a networkX.DiGraph which can be used to visualize the tree the MCTS built. Its parameters are as follows:

  * root_node: node that holds root node
  * function: heuristic function to use in MCTS
  * param: float to use with MCTS to use with UCT
  * end_time: float to use with MCTS to dictate stop time in seconds

The "grapher" function located in the src/mcts_module file can be used to visualize but currently needs improvements 

## TODOs ordered by priority

1. Find way to dynamically set time and UCT parameters based on problem type and size to avoid required external input
2. Start comparisons with Rustiq  - (currently working on this)
3. Try to create more efficient heuristic - (currently working on this)
4. Go over functions in every module and try to design more efficient versions
5. Create better graphing system
