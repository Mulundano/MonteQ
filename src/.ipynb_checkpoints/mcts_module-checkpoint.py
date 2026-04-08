import math
import heapq as hp
import rustworkx as rx
from src.classes import Node, Cirq_Tableau
from qiskit.quantum_info import Clifford
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes.synthesis.hls_plugins import  GreedySynthesisClifford
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import transpile
import time



############################################################
#DAG implementation functions

# Function to create an anticommute DAG from a set of Pauli strings
def build_anticommute_dag(Pauli_word):

    # initialization of graph
    graph = rx.PyDAG()
    graph.add_nodes_from(range(len(Pauli_word)))

    # for loop to create anticommutation connections
    for j in range(len(Pauli_word)):
        for i in range(0, j):
            if not pauli_strings_commute(Pauli_word[i], Pauli_word[j]):
                graph.add_edge(i, j, None)

    return graph

# Function to return indexes of nodes without incoming connections 
def front_layer(graph):

    pred = graph.predecessor_indices
    def func(node):
        return len(pred(node)) == 0

    return graph.filter_nodes(func)
    
###############################################
#MCTS related functions

#Function to count number of CXs present
def cx_count(action_list):
    count = 0
    for action in action_list:
        if action[0] == "CX":
            count += 1

    return count

# function to select a leaf node and expand out non-leaf nodes
def choose(node, function, parameter, kwargs):
    

    # condition checks if there are any untouched actions for a node
    if node.untouched:
        ndx =  node.untouched[0] #rnd.choice(node.untouched) #node.untouched[0] chooses index randomly from remaining untouched indexes
        start = time.time()
        new_state, new_action, new_ndx_list = function(node.state, ndx, node.ndx_list,node.untouched, kwargs) # creates new Pauli word 
        end = time.time()

        # choosing leaf node and filling out its properties
        selected_node = Node(new_state)
        
        if node.dag != None:
            rmv = [i for i in node.ndx_list if i not in new_ndx_list]
            dag = node.dag.copy()
            dag.remove_nodes_from(rmv)
            selected_node.dag = dag
            selected_node.untouched = [new_ndx_list.index(i) for i in front_layer(dag)]
            
        selected_node.parent = node
        selected_node.action = (node.action + new_action)
        selected_node.ndx_list = new_ndx_list
        selected_node.action_time = node.action_time + (end-start)
        node.children.append(selected_node)
        node.untouched = [x for x in node.untouched if x != ndx]
        return selected_node
    else:
        value = -float('inf')
        selected_node = None 

        # loop to go over all the child nodes and pick the one with the highest UCT
        for child in node.children:
            new_val = child.UCT(parameter)
            if new_val > value:
                value = new_val
                selected_node = child

        return selected_node

# function that runs through selections and expansions until it arrives at a leaf (unexpanded) node
def tree_policy(node, function, parameter, kwargs):
    
    # loop to keep selecting and expanding
    while True:

        # if it has no children - leaf node
        if not node.children:

            # if node has been visited or is a root node
            if node.ni != 0 or node.parent == None:
                if node.state.row_num == 0:
                    return node
                    
                
                node = choose(node, function, parameter, kwargs)
            else:
                return node
        else:
            node = choose(node, function, parameter, kwargs) # select one of the children


# function to simulate continuation of tree until a terminal node
def rollout_policy(node, function, solution_list, kwargs):
    
    state = node.state.copy()
    action = node.action.copy()
    action_time = node.action_time
    ndx_list = node.ndx_list.copy()
    allowed = node.untouched.copy()
    dag = None
    if node.dag != None:
        dag = node.dag.copy()
        
    # loop to run until terminal node is found
    while True:

        # break condition
        if state.row_num == 0:
            count = cx_count(action)
            solution = (count,action_time ,len(action),action)
            #print(solution)
            if solution not in solution_list:
                solution_list.append(solution)
            return count
            
        weight_word = state.xs | state.zs
        weight = float('inf')
        index = None

        # loop to find Pauli string in current Pauli word with highest Pauli weight
        #print(weight_word)
        for i, string in enumerate(weight_word):
            w_sum = sum(string)
            if w_sum < weight and i in allowed:
                weight = w_sum 
                index = i
        #index = rnd.choice(range(state.row_num))
        start = time.time()
        new_state, actions, new_ndx_list = function(state, index, ndx_list, allowed, kwargs)
        end = time.time()
        if dag != None:
            rmv = [i for i in ndx_list if i not in new_ndx_list]
            new_dag = dag.copy()
            new_dag.remove_nodes_from(rmv)
            dag = new_dag
            allowed = [new_ndx_list.index(i) for i in front_layer(dag)]
        else:
            allowed = list(range(new_state.row_num))

        state = new_state
        ndx_list = new_ndx_list
        action_time += (end-start)
        # print(index)
        # print(state)
        # print(allowed)
        # print(index)
        #print(state.return_string())
        
        action += actions
        
        


# function to backpropagate estimated value at terminal state up through tree
def backpropagate(node, value):

    # loop to backpropagate upwards until root is found
    while node != None:

        node.qi += value 
        node.ni += 1
        node = node.parent

def best_solution(solution_list):
    hp.heapify(solution_list)
    return solution_list[0][-1]


# Monte Carlo Tree Search Function
def MCTS(root_node, function, param,stop_time, sims,rot_params,checks, kwargs):
    
    '''
    Function that completely executes MCTS
    root_node: node that holds root node
    '''
    start_time = time.time() 
    current_time = start_time
    solution_list = []
    #count = 0
    # loop to run specificed number of iterations
    if stop_time != None and sims == None:
        while current_time < (start_time + stop_time):
            leaf_node = tree_policy(root_node, function, param, kwargs) # selection and expansion phases
            value = rollout_policy(leaf_node, function, solution_list, kwargs) # simulation of leaf node
            backpropagate(leaf_node, value)
            current_time = time.time()
    else:
        if sims == None:
            sims = len(root_node.untouched)
            
        while root_node.ni < sims:
            leaf_node = tree_policy(root_node, function, param, kwargs) # selection and expansion phases
            value = rollout_policy(leaf_node, function, solution_list, kwargs) # simulation of leaf node
            backpropagate(leaf_node, value)

    new_sols = []
    for solution in solution_list:
        action_time = solution[1]
        head = solution[-1]
        tail = [x for x in head if x[0] not in checks]
        tail.reverse()
        for op in tail:
            ndx = tail.index(op)
            if op[0] == "S":
                tail[ndx] = ("S*", op[1])

        cirq, new_order = convert_to_circuit(root_node.state.column_num, head, rot_params)
        tail, order = convert_to_circuit(root_node.state.column_num, tail, rot_params)

        new_tail = GreedySynthesisClifford().run(
            Clifford(tail)
        )

        new_tail = transpile(new_tail, basis_gates = ["cx", "rz", "s", "sdg", "h"], optimization_level=0)
        cirq.compose(new_tail, inplace=True)
        new_sols.append((cirq.count_ops()['cx'], cirq.depth(), cirq.size(), action_time, cirq))
        
    
    solution =   best_solution(new_sols)  
    return solution


#######################################################################################
# Functions going forward do not affect MCTS implementation but use it 

# function to retunr full circuit using MCTS
def full_circuit(Paulis, function, mcts_param, stop_time = None, sims = None, rot_params = None, gate_cancellation=True, order_preserving = True, **kwargs):
    checks = ["-X", "X", "-Y", "Y", "-Z", "Z"]
    
    
    num_paulis = len(Paulis)
    num_qubits = len(Paulis[0])
    
    ndxs = list(range(num_paulis))
    
    
    
      
    indices = ndxs.copy()
    if rot_params == None:
        rot_params = ParameterVector("φ", num_paulis)

    if order_preserving:
        DAG = build_anticommute_dag(Paulis)
        

    root_node = Node(Cirq_Tableau(Paulis))
    root_node.ndx_list = ndxs
    if order_preserving:
        root_node.dag = DAG
        root_node.untouched = [ndxs.index(j) for j in front_layer(DAG)]

    if stop_time != None:
        solution = MCTS(root_node, function, mcts_param,(tableau.row_num/num_paulis)*stop_time, sims, rot_params, checks,kwargs)
    else:
        solution = MCTS(root_node, function, mcts_param, None, sims,rot_params, checks,kwargs)
    
   
    return solution


#############################################33
# miscellanous fnctions that are used in this module or can be used 

def convert_to_circuit(num, solution, rot_params):
    
    pos = (num -1)
    qc = QuantumCircuit(num)
    new_order = []
    for op in solution:
        
        match op[0]:
            case "CX":
                qc.cx(op[1],op[2])
            case "S":
                qc.s(op[1])
            case "S*":
                qc.sdg(op[1])
            case "H":
                qc.h(op[1])
            case "-X" | "X":
                new_order.append(op[2])
                qc.h(op[1])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),op[1])
                else:
                    qc.rz(2*rot_params[op[2]],op[1])
                qc.h(op[1])

            case "-Y" | "Y":
                new_order.append(op[2])
                qc.s(op[1])
                qc.h(op[1])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),op[1])
                else:
                    qc.rz(2*rot_params[op[2]],op[1])
                qc.h(op[1])
                qc.sdg(op[1])

            case "-Z" | "Z":
                new_order.append(op[2])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),op[1])
                else:
                    qc.rz(2*rot_params[op[2]],op[1])

    return qc, new_order

def pauli_strings_commute(pauli_str1, pauli_str2):
    """
    Determine if two Pauli strings commute.
    
    pauli_str1: A str representing the first Pauli operator.
    pauli_str2: A str representing the second Pauli operator.
    return: True if the Pauli strings commute, False otherwise.
    -referenced from QuCLEAR code-
    """
    if len(pauli_str1) != len(pauli_str2):
        raise ValueError("Pauli strings must be of the same length.", pauli_str1, pauli_str2)
    
    commute = True  # Assume they commute until proven otherwise
    
    anticommute_count = 0
    
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != pauli_str2[i] and pauli_str1[i] != 'I' and pauli_str2[i] != 'I':
            # Found anti-commuting Pauli matrices
            commute = False
            anticommute_count += 1

    if anticommute_count % 2 == 0:
        commute = True
    
    return commute