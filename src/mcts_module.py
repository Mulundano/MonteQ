import math
from src.classes import Node, Cirq_Tableau
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.converters import circuit_to_dag, dag_to_circuit
import matplotlib.pyplot as plt
import time

def cx_count(action_list):
    '''
    Function to count number of CXs present

    action_list: List containing information on gates applied 
    '''
    count = 0
    for action in action_list:
        if action[0] == "CX":
            count += 1

    return count


def choose(node, function, parameter):
    '''
    Function to select a leaf node and expand out non-leaf nodes

    node: Node to begin process with 
    function: heuristic function used to expand out tree
    parameter: float value for UCT 
    '''

    # condition checks if there are any untouched actions for a node
    if node.untouched:
        ndx = rnd.choice(node.untouched) #node.untouched[0] chooses index randomly from remaining untouched indexes
        new_state, new_action, new_ndx_list = function(node.state, ndx, node.ndx_list) # creates new Pauli word 

        # condition to check if current Pauli word is fully implemented but have list of Pauli words in set has not been exhausted
        if new_state.row_num == 0 and node.plist:
            new_state = node.plist[0].copy()
            actions = node.action + new_action

            # loop to have actions applied to next Pauli word
            for op in actions:
                match op[0]:
                    case "CX":
                        new_state.apply_CX(op[1], op[2])
                    case "S":
                        new_state.apply_S(op[1])
                    case "H":
                        new_state.apply_H(op[1])

            # choosing leaf node and filling out its properties
            selected_node = Node(new_state)
            selected_node.parent = node
            selected_node.action = actions
            selected_node.ndx_list = new_ndx_list
            selected_node.plist = [x for x in node.plist if x != node.plist[0]]
            node.children.append(selected_node)
            node.untouched = [x for x in node.untouched if x != ndx]
            return selected_node

        # choosing leaf node and filling out its properties
        selected_node = Node(new_state)
        selected_node.parent = node
        selected_node.plist = [x.copy() for x in node.plist]
        selected_node.action = (node.action + new_action)
        selected_node.ndx_list = new_ndx_list
        node.children.append(selected_node)
        node.untouched = [x for x in node.untouched if x != ndx]
        return selected_node
    else:
        value = float('inf')
        selected_node = None 

        # loop to go over all the child nodes and pick the one with the highest UCT
        for child in node.children:
            new_val = child.UCT(parameter)
            if new_val < value:
                value = new_val
                selected_node = child

        return selected_node


def tree_policy(node, function, parameter):
    '''
    Function that runs through selections and expansions until it arrives at a leaf (unexpanded) node

    node: selected node to expand out
    function: heuristic function to prduce action
    parameter: float to hold paramter for UCT calculations
    '''

    # loop to keep selecting and expanding
    while True:

        # if it has no children - leaf node
        if not node.children:

            # if node has been visited or is a root node
            if node.ni != 0 or node.parent == None:
                if node.state.row_num == 0:
                    return node
                    
                
                node = choose(node, function, parameter)
            else:
                return node
        else:
            node = choose(node, function, parameter) # select one of the children



def rollout_policy(node, function, solution_list):
    '''
    Function to simulate continuation of tree until a terminal node

    node: node to simulate from
    function: heuristic function to produce action between nodes
    '''
    state = node.state.copy()
    action = node.action.copy()
    plist = node.plist.copy()
    ndx_list = node.ndx_list.copy()
    # loop to run until terminal node is found
    while True:

        # break condition
        if state.row_num == 0:
            count = cx_count(action)
            solution = (count, action)
            #print(solution)
            if solution not in solution_list:
                solution_list.append(solution)
            return count
            
        weight_word = state.xs | state.zs
        weight = float('inf')
        index = None

        # loop to find Pauli string in current Pauli word with highest Pauli weight
        for i, string in enumerate(weight_word):
            w_sum = sum(string)
            if w_sum < weight:
                weight = w_sum 
                index = i
        #index = rnd.choice(range(state.row_num))
        state, actions, ndx_list = function(state, index, ndx_list)
        if state.row_num == 0 and plist:
            state = plist[0]
            plist = [x for x in plist if x != state]
            #ndx_list = list(range(state.row_num))
        action += actions
        



def backpropagate(node, value):
    '''
    Function to backpropagate estimated value at terminal state up through tree

    node: current node to backpropagate from
    value: value at terminal state
    '''

    # loop to backpropagate upwards until root is found
    while node != None:

        node.qi += value 
        node.ni += 1
        node = node.parent

def best_solution(solution_list):
    hp.heapify(solution_list)
    return solution_list[0][1]



def MCTS(root_node, function, param, timeout):
    '''
    Function that completely executes MCTS
    root_node: node that holds root node
    '''
    start_time = time.time() 
    current_time = start_time
    solution_list = []
    # count = 0
    # loop to run specificed number of iterations
    while current_time < start_time + timeout:
        leaf_node = tree_policy(root_node, function, param) # selection and expansion phases 
        value = rollout_policy(leaf_node, function, solution_list) # simulation of leaf node
        backpropagate(leaf_node, value)
        current_time = time.time()
        # count += 1
    # print(count)
    solution =   best_solution(solution_list)  
    return solution


#######################################################################################
# Functions going forward do not affect MCTS implementation but use it 

def p_word_solution(p_word, function, mcts_param, end_time):
    '''
    Function to create a solution for a Pauli word 

    p_word: Pauli word in Tableau form
    function: heuristic function to use in MCTS 
    mcts_param: float to use with MCTS to use with UCT
    end_time: float to use with MCTS to dictate stop time in seconds
    '''
    path = []

    # loop to run MCTS until the Pauli word is fully implemented
    while len(p_word.xs) != 0:
        action, new_word, graph = MCTS(Node(p_word), function, mcts_param, end_time)
        path += action
        p_word = new_word
        

    return path # returns a list containing the 'head' of the Pauli word circuit for a single Pauli word



def full_circuit(p_sentence, function, mcts_param, stop_time, divs=3, rot_params = None, gate_cancellation=True, synth = 1):
    tail = []
    head = []
    num_paulis = sum([len(x) for x in p_sentence])
    num_qubits = len(p_sentence[0][0])
    ndx_list = list(range(num_paulis))  
    indices = ndx_list.copy()
    if rot_params == None:
        rot_params = ParameterVector("φ", num_paulis)
        
    paulis = ["-X", "X", "-Y", "Y", "-Z", "Z"]
    p_word_group = [p_sentence[i:i + divs] for i in range(0, len(p_sentence), divs)]
    
    for word_group in p_word_group:
        word_group = [Cirq_Tableau(word) for word in word_group]
        root = word_group[0]
        
        
        if head:
            for op in head:
                match op[0]:
                    case "CX":
                        root.apply_CX(op[1], op[2])
                    case "S":
                        root.apply_S(op[1])
                    case "H":
                        root.apply_H(op[1])
                        
            root_node = Node(root)
            root_node.plist = [x for x in word_group if x != root]
            root_node.ndx_list = indices
            solution = MCTS(root_node, function, mcts_param, stop_time)
            head += solution
            #print(head)
            tail += [x for x in solution if x[0] not in paulis]
            rmv_paulis = len(head) - len(tail)
            indices = ndx_list[rmv_paulis:]
        else:
            root_node = Node(root)
            root_node.plist = [x for x in word_group if x != root]
            root_node.ndx_list = indices
            solution = MCTS(root_node, function, mcts_param, stop_time)  
            head += solution
            #print(head)
            tail += [x for x in solution if x[0] not in paulis]
            rmv_paulis = len(head) - len(tail)
            indices = ndx_list[rmv_paulis:]

    tail.reverse()
    for op in tail:
        ndx = tail.index(op)
        if op[0] == "S":
            tail[ndx] = ("S*", op[1])
    #head += tail

    cirq, new_order = convert_to_circuit(num_qubits, head, rot_params)
    tail, order = convert_to_circuit(num_qubits, tail, rot_params)
    
    if synth == 0:
        cirq.compose(tail, inplace=True)
    else:
        new_tail = GreedySynthesisClifford().run(
            Clifford(tail)
        )
        cirq.compose(new_tail, inplace=True)
        
    if gate_cancellation:
        cirq = dag_to_circuit(
            CommutativeCancellation().run(
                circuit_to_dag(cirq)
            )
        )

    return cirq, new_order


#############################################33
# miscellanous fnctions that are used in this module or can be used 

def convert_to_circuit(num, solution, rot_params):
    
    pos = (num -1)
    qc = QuantumCircuit(num)
    new_order = []
    for op in solution:
        
        match op[0]:
            case "CX":
                qc.cx(pos - op[1],pos - op[2])
            case "S":
                qc.s(pos - op[1])
            case "S*":
                qc.sdg(pos-op[1])
            case "H":
                qc.h(pos-op[1])
            case "-X" | "X":
                new_order.append(op[2])
                qc.h(pos-op[1])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),pos-op[1])
                else:
                    qc.rz(2*rot_params[op[2]],pos-op[1])
                qc.h(pos-op[1])

            case "-Y" | "Y":
                new_order.append(op[2])
                qc.s(pos - op[1])
                qc.h(pos-op[1])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),pos-op[1])
                else:
                    qc.rz(2*rot_params[op[2]],pos-op[1])
                qc.h(pos-op[1])
                qc.sdg(pos - op[1])

            case "-Z" | "Z":
                new_order.append(op[2])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[op[2]]),pos-op[1])
                else:
                    qc.rz(2*rot_params[op[2]],pos-op[1])

    return qc, new_order