import math
from src.classes import Node, Cirq_Tableau
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.converters import circuit_to_dag, dag_to_circuit
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
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


#TODO: make graphing feature properly useful
def expand(node, function, graph):
    '''
    Function to expand out children of a leaf node

    node: selected node to expand out
    function: heuristic function to produce action
    '''
    
    # loop to create action for each possible child
    for i in range(node.state.row_num):
        new_state, new_action = function(node.state, i)
        graph.add_edge(f"[{node.state.return_string()}]", f"[{new_state.return_string()}]")
        new_node = Node(new_state)
        new_node.action = new_action
        new_node.parent = node
        new_node.curr_cx_num = node.curr_cx_num + cx_count(new_action)
        if new_node not in node.children:
            node.children.append(new_node)


def select(node, parameter):
    '''
    Function to select one of the children of a node

    node: selected node to expand out
    parameter: float to use in UCT
    '''
    
    value = -float('inf')
    selected_node = None 

    # loop to go over all the child nodes and pick the one with the highest UCT
    for child in node.children:
        new_val = child.UCT(parameter)
        if new_val > value:
            value = new_val
            selected_node = child

    return selected_node


def tree_policy(node, function, parameter, tree):
    '''
    Function that runs through selections and expansions until it arrives at a leaf (unexpanded) node

    node: selected node to expand out
    function: heuristic function to prduce action
    parameter: float to hold parameter for UCT calculations
    '''

    # loop to keep selecting and expanding
    while True:

        # if it has no children - leaf node
        if not node.children:

            # if node has been visited or is a root node
            if node.ni != 0 or node.parent == None:
                if node.state.row_num == 0:
                    return node
                    
                expand(node, function, tree)
                node = select(node, parameter)
            else:
                return node
        else:
            node = select(node, parameter) # select one of the children



def rollout_policy(node, function):
    '''
    Function to simulate continuation of tree until a terminal node

    node: node to simulate from
    function: heuristic function to produce action between nodes
    '''
    state = node.state
    cx_num = node.curr_cx_num

    # loop to run until terminal node is found
    while True:

        # break condition
        if state.row_num == 0:
            return cx_num
            
        weight_word = state.xs | state.zs
        weight = -float('inf')
        index = None

        # loop to find Pauli string in current Pauli word with highest Pauli weight
        for i, string in enumerate(weight_word):
            w_sum = sum(string)
            if w_sum > weight:
                weight = w_sum 
                index = i

        state, actions = function(state, index)
        cx_num += cx_count(actions)



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



def best_action(root_node):
    '''
    Function to acquire the best action to take based on average cx count across simulations
    '''
    path = None 
    num = float('inf')
    node = None
    # print(root_node.state.return_string())
    # print([(x.state.return_string(), x.action) for x in root_node.children])
    # loop to go through all child nodes and pick the one with lowest cx count
    for child in root_node.children:
        
        if child.qi != 0:
            if (child.qi/child.ni) < num:
                num = (child.qi/child.ni)
                path = child.action
                node = child
            elif (child.qi/child.ni) == num and child.ni > node.ni:
                num = (child.qi/child.ni)
                path = child.action
                node = child
    #print(path)    
    return path, node.state


def MCTS(root_node, function, param, timeout):
    '''
    Function that completely executes MCTS
    root_node: node that holds root node
    '''
    
    tree_graph = nx.DiGraph()# graph to store tree produced in MCTS for visualizations

    start_time = time.time() 
    current_time = start_time
    # loop to run specificed number of iterations
    while current_time < start_time + timeout:
        leaf_node = tree_policy(root_node, function, param, tree_graph) # selection and expansion phases 
        value = rollout_policy(leaf_node, function) # simulation of leaf node
        backpropagate(leaf_node, value)
        current_time = time.time()
    action, state =   best_action(root_node)  
    return action, state, tree_graph


#######################################################################################
# Functions going forward do not affect MCTS implementation but use it 

def p_word_solution(p_word, function, mcts_param, word_time):
    '''
    Function to create a solution for a Pauli word 

    p_word: Pauli word in Tableau form
    function: heuristic function to use in MCTS 
    mcts_param: float to use with MCTS to use with UCT
    end_time: float to use with MCTS to dictate stop time in seconds
    '''
    path = []
    order = list(range(p_word.row_num))
    # loop to run MCTS until the Pauli word is fully implemented
    while p_word.row_num != 0:
        #move_time = word_time/p_word.row_num
        #print(f"Move time {move_time}s")
        action, new_word, graph = MCTS(Node(p_word), function, mcts_param, word_time)
        path += action
        p_word = new_word
        
        #grapher(graph)

    return path # returns a list containing the 'head' of the Pauli word circuit for a single Pauli word



def full_circuit(p_sentence, function, mcts_param, total_time, rot_params = None, gate_cancellation=False):
    '''
    Function to construct full circuit given a set of Pauli words where each Pauli word is composed of mutually commuting Pauli strings
    
    p_sentence: list of Pauli words where each is given as a list of Pauli strings
    function: heuristic function to use in MCTS 
    stop_time: float to use with MCTS to dictate stop time in seconds
    rot_params: list of floats which are the rotation parameters to use for each Pauli string - if None placeholders are made
    gate_cancellation: boolean value to dictate whether to apply CommutativeCancellation from Qiskit (removes gates that cancel)
    '''
    
    tail = []
    head = []
    num_paulis = sum(len(x) for x in p_sentence)
    num_qubits = len(p_sentence[0][0])

    # condition to create place hoolder rotation parameters
    if rot_params == None:
        rot_params = ParameterVector("φ", num_paulis)
        
    paulis = ["-X", "X", "-Y", "Y", "-Z", "Z"]

    #word_times = [total_time*(x/totals_sum) for x in totals]
    # loop to solve each Pauli word in the list
    for commute_word in p_sentence:
        p_word = Cirq_Tableau(commute_word) # turns a list of Pauli strings into a Tableau

        
        #word_time = (p_word.row_num/num_paulis)*total_time
        #print(f"Word time {word_time}s")
        if head:
            for op in head:
                match op[0]:
                    case "CX":
                        p_word.apply_CX(op[1], op[2])
                    case "S":
                        p_word.apply_S(op[1])
                    case "H":
                        p_word.apply_H(op[1])
                        
            if implement_checker(p_word, commute_word, head):
                continue
            solution = p_word_solution(p_word, function, mcts_param, total_time)
            head += solution
            tail += [x for x in solution if x[0] not in paulis]
        else:
            if implement_checker(p_word, commute_word, head):
                continue
            solution = p_word_solution(p_word, function, mcts_param, total_time)  
            head += solution
            tail += [x for x in solution if x[0] not in paulis]

    tail.reverse()
    for op in tail:
        ndx = tail.index(op)
        if op[0] == "S":
            tail[ndx] = ("S*", op[1])
    head += tail

    cirq = convert_to_circuit(num_qubits, head, rot_params)
    if gate_cancellation:
        cirq = dag_to_circuit(
            CommutativeCancellation().run(
                circuit_to_dag(cirq)
            )
        )

    return cirq


#############################################33
# miscellanous fnctions that are used in this module or can be used 
def implement_checker(p_word, commute_word,  head):
    implemented = False
    if p_word.row_num == 1:
            x_z = p_word.xs[0] | p_word.zs[0]
            in_ndx = 0
            if sum(x_z) == 1:
                implemented = True
                for ndx, pauli in enumerate(x_z):
                    if pauli == 1:
                        in_ndx = ndx
                        break

                if "-" in commute_word:
                    if "X" in commute_word:
                        head.append(("-X", in_ndx))
                    elif "Y" in commute_word:
                        head.append(("-Y", in_ndx))
                    elif "Z" in commute_word:
                        head.append(("-Z", in_ndx))
                        
                else:
                    if "X" in commute_word:
                        head.append(("X", in_ndx))
                      
                    elif "Y" in commute_word:
                        head.append(("Y", in_ndx))
                        
                    elif "Z" in commute_word:
                        head.append(("Z", in_ndx))
    return implemented

def convert_to_circuit(num, solution, rot_params):
    '''
    Function to convert a given solution into a Qiskit QuantumCircuit

    num: int storing the number of qubits
    rot_params: list of floats which are the rotation parameters to use for each Pauli string
    '''
    pos = (num -1)
    ndx = 0
    qc = QuantumCircuit(num)

    # loop that updates the QuantumCircuit with each operation
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
                qc.h(pos-op[1])
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[ndx]),pos-op[1])#math.pi + rot_params[ndx]),pos-op[1])
                else:
                    qc.rz(2*rot_params[ndx],pos-op[1])
                qc.h(pos-op[1])
                ndx += 1
            case "-Y" | "Y":
                qc.s(pos - op[1])
                qc.h(pos-op[1])
                if "-" in op[0]:
                    qc.rz(2*rot_params[ndx],pos-op[1])#(2*(math.pi + rot_params[ndx]),pos-op[1])
                else:
                    qc.rz(2*(-rot_params[ndx]),pos-op[1])
                qc.h(pos-op[1])
                qc.sdg(pos - op[1])
                ndx += 1
            case "-Z" | "Z":
                if "-" in op[0]:
                    qc.rz(2*(-rot_params[ndx]),pos-op[1])#(2*(math.pi + rot_params[ndx]),pos-op[1])
                else:
                    qc.rz(2*rot_params[ndx],pos-op[1])
                ndx += 1
    return qc


#TODO: make better grapher program
def grapher(graph):
    '''
    Simple function to graph out tree produced in MCTS - need to improve

    graph: directed graph that represents the tree produced by MCTS
    '''
    plt.figure(figsize=(35, 30))
    nx.nx_agraph.write_dot(graph,'test.dot')
    pos =graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos,  with_labels = True, node_color="white", node_size=3500)