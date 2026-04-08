import numpy as np
from src.classes import Cirq_Tableau
from src.mcts_module import cx_count
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeAlgiers, FakeAlmadenV2, FakeManhattanV2
from qiskit.transpiler.passes import DenseLayout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit

# function to remove rows in a tableau with only a single nonidentity operation
def prune(tableau: Cirq_Tableau, ndx_list, allowed):
    
    prn_ndxs = [] # list to hold all row indices to be pruned
    
    tab = tableau.copy() # copy of tableau to work on

    single_q = [] # loist of single qubit operations 
    
    track_ndx = ndx_list.copy()
    
    rmv_ndx = []
    # extracts x, z and sign arrays
    x, z, s = tab.xs, tab.zs, tab.ss

    # bitwise ORs x and z arrays in tableau to create array where nonidentity operation indices have 1 in them
    weight_array = x | z

    # loop that checks through each row to find out if it has a Pauli weight of 1
    for r_ndx in range(len(weight_array)):
        if sum(weight_array[r_ndx])  == 1 and r_ndx in allowed:
            val = track_ndx[r_ndx]
            # appends index to be pruned if Pauli weight == 1
            prn_ndxs.append(r_ndx)
            rmv_ndx.append(val)
            # checks what type of Pauli is on that index X, Y or Z and appends to path with action weight of zero 
            if sum(z[r_ndx]) == 0:
                if s[r_ndx] == 0:
                    single_q.append(("X", int(np.argmax(x[r_ndx] == 1)), val))
                else:
                    single_q.append(("-X", int(np.argmax(x[r_ndx] == 1)), val))
            elif sum(x[r_ndx]) == 0:
                if s[r_ndx] == 0:
                    single_q.append(("Z", int(np.argmax(z[r_ndx] == 1)), val))
                else:
                    single_q.append(("-Z", int(np.argmax(z[r_ndx] == 1)), val))
            else:
                if s[r_ndx] == 0:
                    single_q.append(("Y", int(np.argmax(x[r_ndx] == 1)), val))
                else:
                    single_q.append(("-Y", int(np.argmax(x[r_ndx] == 1)), val))
                    
        elif sum(weight_array[r_ndx])  == 0: # elif removes fully I lines 
            x, z, s = np.delete(x, r_ndx, axis=0), np.delete(z, r_ndx, axis=0), np.delete(s, r_ndx)
            
    # prunes rows 
    x, z, s = np.delete(x, prn_ndxs, axis=0), np.delete(z, prn_ndxs, axis=0), np.delete(s, prn_ndxs)
    column_num = len(x[0]) if x.size != 0 else 0 
    row_num = len(x)
    new_ndx_list = [x for x in ndx_list if x not in rmv_ndx]
    # updates tableau and returns it 
    tab.xs, tab.zs, tab.ss, tab.column_num, tab.row_num = x, z, s, column_num,row_num
    return tab, single_q, new_ndx_list
    

#Function to identify the best operation amongst a list - This function is mainly used to identify which one of the allowed transformations for a pair produces the best benefit
def identify(tableau, op_list, ctrl, targ):
    
    operation = None # best operation
    benefit = -float('inf') # benefit of an operation benefit = reduce - increase 
    reduced = 0 # number of reducible pairs across the ctrl and targ (equivalent to how many single qubit operations will be removed)
    increased = 0 # number of increasing pairs across the ctrl and targ (equivalent to how many single qubit operations will be added)

    # loop to run through possible operations 
    for op in op_list:
        
        tab = tableau.copy()

        # loop to go over all the gates in the operation and apply them accordingly
        for act in op:
            match act[0]:
                case "H":
                    tab.apply_H(act[1])
                case "S":
                    tab.apply_S(act[1])

        # calculation of reduced and increased for this particular action
        reduce = sum((tab.xs[:, ctrl] & tab.xs[:, targ] & ~tab.zs[:, targ]) | (~tab.xs[:, ctrl] & tab.zs[:, ctrl] & tab.zs[:, targ]))
        increase = sum((~tab.xs[:, ctrl] & ~tab.zs[:, ctrl] & tab.zs[:, targ]) | (tab.xs[:, ctrl] & ~tab.xs[:, targ] & ~tab.zs[:, targ]))
        
        # condition to calculate better benefit and replace necessary values
        # benefit is the total no. of reducible pairs minus total no. of increasing pairs
        # this will total how many single qubits will be removed from the entire Pauli word
        if reduce - increase > benefit:
            benefit = reduce - increase
            operation = op
            reduced = reduce
            increased = increased
        elif reduce - increase == benefit:
            if reduce > reduced:
                benefit = reduce - increase
                operation = op
                reduced = reduce
                increased = increase

    # appending CX to the operation to complete it
    operation.append(("CX", ctrl, targ))
    return reduced, increased, operation 

# Function that takes in a pair and identifies what Paulis are in it
def compare(tab, ndx, ctrl, targ):
    

    if tab.xs[ndx][ctrl] & ~tab.zs[ndx][ctrl]: # if control is X
        if tab.xs[ndx][targ] & ~tab.zs[ndx][targ]: # if target is X

            # list of allowed transformations for XX
            op_list = [[], [("S", ctrl)], [("H", ctrl), ("H", targ)],
                      [("H", ctrl), ("S", targ)]]
            return identify(tab, op_list, ctrl, targ)
            
        elif tab.xs[ndx][targ] & tab.zs[ndx][targ]: # if target is Y

            # list of allowed transformations for XY
            op_list = [[("S", ctrl), ("S", targ)], [("S", targ)], [("H", ctrl)],
                      [("H", ctrl), ("H", targ)]]
            return identify(tab, op_list, ctrl, targ)

        else:# if target is Z

            # list of allowed transformations for XZ
            op_list = [[("S", ctrl), ("H", targ)], [("H", targ)], [("H", ctrl)],
                      [("H", ctrl), ("S", targ)]]
            return identify(tab, op_list, ctrl, targ)
            
    elif tab.xs[ndx][ctrl] & tab.zs[ndx][ctrl]: # if control is Y
        if tab.xs[ndx][targ] & ~tab.zs[ndx][targ]: # if target is X

            # list of allowed transformations for YX
            op_list = [[], [("S", ctrl)], [("H", ctrl)]]
            return identify(tab, op_list, ctrl, targ)
            
        elif tab.xs[ndx][targ] & tab.zs[ndx][targ]: # if target is Y

            # list of allowed transformations for YY
            op_list = [[("S", ctrl), ("S", targ)], [("H", ctrl),("S", targ)],
                      [("S", targ)]]
            return identify(tab, op_list, ctrl, targ)

        else:# if target is Z

            # list of allowed transformations for YZ
            op_list = [[("H", targ)], [("S", ctrl), ("H", targ)]]
            return identify(tab, op_list, ctrl, targ)
            
    else: # if control is Z
        if tab.xs[ndx][targ] & ~tab.zs[ndx][targ]: # if target is X

            # list of allowed transformations for ZX
            op_list = [[("H", targ)], [("H", ctrl)], [("S", ctrl),("H", targ)],
                      [("S", targ)]]
            return identify(tab, op_list, ctrl, targ)
            
        elif tab.xs[ndx][targ] & tab.zs[ndx][targ]: # if target is Y

            # list of allowed transformations for ZY
            op_list = [[], [("S", ctrl)], [("H", targ)], [("S", ctrl),("H", targ)], [("H", ctrl), ("S", targ)]]
            return identify(tab, op_list, ctrl, targ)

        else:# if target is Z

            # list of allowed transformations for ZZ
            op_list = [[], [("H", ctrl), ("H", targ)], [("S", ctrl)], [("S", targ)], [("S", ctrl), ("S", targ)]]
            return identify(tab, op_list, ctrl, targ)

# Heuristic function to produce a single reduction from one node of a tree to a child
def pair_solve(tableau, ndx, ndx_list,allowed, kwargs):
    
    tab = tableau.copy()
    reduction = [] # reduction of Pauli string

    # loop to run until break condition is met - Pauli string is implementable
    while True:
        operation = None
        benefit = -float('inf')
        reduced = 0
        increased = 0

        # break condition 
        if sum(tab.xs[ndx] | tab.zs[ndx]) == 1:
            break
            
        #  loop to cycle over all qubit indexes to get all combinations of control and target 
        for i in range(tab.column_num):
            if i != tab.column_num -1 and (tab.xs[ndx][i] or tab.zs[ndx][i]): # prevents I's 
                for j in range(i+1, tab.column_num):
                    if tab.xs[ndx][j] or tab.zs[ndx][j]: # prevents I's 

                        # conditions to check for better benefit with i as ctrl and j as targ
                        reduce, increase, op = compare(tab, ndx, i, j)
                        if (reduce - increase) > benefit:
                            benefit = reduce - increase
                            operation = op
                            reduced = reduce
                            increased = increase
                        elif (reduce - increase) == benefit:
                            if reduce > reduced:
                                operation = op
                                reduced = reduce
                                increased = increase
                                
                        # conditions to check for better benefit with j as ctrl and i as targ
                        reduce, increase, op = compare(tab, ndx, j, i)
                        if (reduce - increase) > benefit:
                            benefit = reduce - increase
                            operation = op
                            reduced = reduce
                            increased = increase
                        elif (reduce - increase) == benefit:
                            if reduce > reduced:
                                operation = op
                                reduced = reduce
                                increased = increase

        # condition that picks the best operation and applies it
        if operation != None:
            for action in operation:
                match action[0]:
                    case "CX":
                        tab.apply_CX(action[1], action[2])
                    case "S":
                        tab.apply_S(action[1])
                    case "H":
                        tab.apply_H(action[1])
                        
            reduction += operation # appends to reduction

    # prunes of implemented Pauli string
    tab, single_q, new_ndx_list = prune(tab, ndx_list, allowed)
    reduction += single_q # appends to reduction
    
    return tab, reduction, new_ndx_list

######################################################################

# function that takes original coupling and reframes it in of the indexes of qubits in the code
def generate_new_couple_list(coupling_list, register, phys_virt_map):
    couple_list = []
    key_list = phys_virt_map.keys()
    for couple in coupling_list:
        if couple[0] in key_list and couple[1] in key_list:
            couple_list.append([register.index(phys_virt_map[couple[0]]), register.index(phys_virt_map[couple[1]])])

    return couple_list

# gievn some IBM layout algorithm, this function returns the the coupling list in terms of the code, the distance matrix and the coupling map
def ibm_layout(num_qubits, machine=FakeAlmadenV2, algorithm=DenseLayout): 
    
    backend = machine()
    
    original_coupling_list = backend.configuration().coupling_map
    original_cmap = CouplingMap(original_coupling_list)
    instance = algorithm(coupling_map=original_cmap)
    instance.run(circuit_to_dag(QuantumCircuit(num_qubits)))

    phys_virt_map = instance.property_set["layout"].get_physical_bits()
    register = list(instance.property_set['layout'].get_registers())[0]

    new_coupling_list = generate_new_couple_list(original_coupling_list, register, phys_virt_map)
    distance_matrix = np.array(CouplingMap(new_coupling_list).distance_matrix)

    return distance_matrix, new_coupling_list, CouplingMap(new_coupling_list)

# generates list of opoerations that reduce furthest qubits
def generate_ops(tableau, pauli_ndx, distance_matrix, coupling_list):
    
    tab = tableau.copy()
    ops = []

    # occupancy vector for specific Pauli string we are trying to solve
    occupancy = tab.xs[pauli_ndx] | tab.zs[pauli_ndx]

    # distance metric for single Pauli string
    dist_vals = np.matmul(distance_matrix, occupancy) * occupancy

    # instead of taking sum this is a quick way to find qub
    ndxs = np.where(dist_vals == np.max(dist_vals))[0]
    
    for ndx in ndxs:
        for couple in coupling_list:
            #if (~tab.xs[pauli_ndx][couple[1]] & ~tab.zs[pauli_ndx][couple[1]])& (~tab.xs[pauli_ndx][couple[1]] & ~tab.zs[pauli_ndx][couple[1]])
            op = []
            if couple[0] == ndx: # qubit to be eliminated lies on control
                if  int(not tab.xs[pauli_ndx][couple[1]]) & int(not tab.zs[pauli_ndx][couple[1]]): # target is I
                    ops.append([("CX", int(ndx), couple[1]),("CX", couple[1], int(ndx)),("CX", int(ndx), couple[1])])
                    continue 
                    
                if tab.xs[pauli_ndx][ndx] & ~tab.zs[pauli_ndx][ndx]: #control is X
                    op.append(("H", int(ndx)))
                elif tab.xs[pauli_ndx][ndx] & tab.zs[pauli_ndx][ndx]: #control is Y
                    op.append(("S", int(ndx)))
                    op.append(("H", int(ndx)))
                    

                if tab.xs[pauli_ndx][couple[1]] & ~tab.zs[pauli_ndx][couple[1]]: #target is X
                    ops.append(op + [("H", couple[1]), ("CX", int(ndx), couple[1])])
                    ops.append(op + [("S", couple[1]), ("CX", int(ndx), couple[1])])
                else:
                    ops.append(op + [("CX", int(ndx), couple[1])])

            elif couple[1] == ndx:# qubit to be eliminated lies on target
                if int(not tab.xs[pauli_ndx][couple[0]]) & int( not tab.zs[pauli_ndx][couple[0]]): # control is I
                    ops.append([("CX", couple[0], int(ndx)),("CX", int(ndx), couple[0]),("CX", couple[0], int(ndx))])
                    continue
                
                if tab.xs[pauli_ndx][ndx] & tab.zs[pauli_ndx][ndx]: #target is Y
                    op.append(("S", int(ndx)))
                elif ~tab.xs[pauli_ndx][ndx] & tab.zs[pauli_ndx][ndx]: #target is Z
                    op.append(("H", int(ndx)))
                
                if ~tab.xs[pauli_ndx][couple[0]] & tab.zs[pauli_ndx][couple[0]]: #control is Z
                    ops.append(op + [("H", couple[0]), ("CX", couple[0], int(ndx))])
                    ops.append(op + [("H", couple[0]),("S", couple[0]), ("CX", couple[0], int(ndx))])
                else:
                    ops.append(op + [("CX", couple[0], int(ndx))])
    
    if any(cx_count(x) == 1 for x in ops):
        ops = [x for x in ops if cx_count(x) == 1]
    
    return ops

# selects best operation from given list
def best_oper(tableau,pauli_ndx, op_list, distance_matrix, switch):
    
    tab = None
    operation = None
    distance = float('inf')

    pauli_occupancy = tableau.xs[pauli_ndx] | tableau.zs[pauli_ndx]

    pauli_distance = sum(np.matmul(distance_matrix, pauli_occupancy) * pauli_occupancy)
    
    
    for op in op_list:
        table = tableau.copy()

        for action in op:
            match action[0]:
                case "CX":
                    table.apply_CX(action[1], action[2])
                case "S":
                    table.apply_S(action[1])
                case "H":
                    table.apply_H(action[1])
        
        occupancy = table.xs | table.zs
        p_occupancy = occupancy[pauli_ndx]
        pauli_dist = sum(np.matmul(distance_matrix, p_occupancy) * p_occupancy)

        if switch:
            if pauli_dist <= pauli_distance: 
           
                operation = op             
                tab = table
                pauli_distance = pauli_dist

        else:
            dist_vals = np.matmul(distance_matrix, occupancy.T) * occupancy.T
            dist = sum(sum(dist_vals))
            
            if (dist < distance and pauli_dist <= pauli_distance): 
           
                operation = op
                tab = table
                distance = dist
                pauli_distance = pauli_dist
    return tab, operation

# full hardware aware solving function that takes into account looping when no reduction happens
def hrdwre_awr_solve(tableau, pauli_ndx, ndx_list,allowed,kwargs):

    distance_matrix, coupling_list = kwargs['distance_matrix'], kwargs['coupling_list']
    tab = tableau.copy()
    reduction = []
    elapsed_states = []
    greedy_switch = False # greedy switch in case there is a loop 
    while sum(tab.xs[pauli_ndx] | tab.zs[pauli_ndx]) != 1:
        
        operations = generate_ops(tab, pauli_ndx, distance_matrix, coupling_list)
       
        new_tab, best_op  = best_oper(tab,pauli_ndx, operations, distance_matrix, greedy_switch)

        if new_tab not in elapsed_states:
            tab = new_tab.copy()
            reduction += best_op
            elapsed_states.append(new_tab)
          
            
        else:
            greedy_switch = True
            index = elapsed_states.index(new_tab)
            reduction = reduction[:index+1]
            
            tab = new_tab
            continue
        
        greedy_switch = False
    tab, single_q, new_ndx_list = prune(tab, ndx_list, allowed)
    reduction += single_q
    
    return tab, reduction, new_ndx_list