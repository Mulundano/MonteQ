import numpy as np
from src.classes import Cirq_Tableau


# TODO: write a more efficient pruning algorithm
def prune(tableau: Cirq_Tableau, ndx_list):
    '''
    Function to remove rows in a tableau with only a single nonidentity operation
    
    :param tableau: Cirq_Tableau to act on
    '''
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
        if sum(weight_array[r_ndx])  == 1:
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
    


def identify(tableau, op_list, ctrl, targ):
    '''
    Function to identify the best operation amongst a list 
    This function is mainly used to identify which one of the allowed transformations for a pair produces the best benefit

    tableau: Cirq_Tableau that holds our Pauli word
    op_list: list of action tuples which are possible operations
    ctrl: int that is index of control qubit
    targ: int that is index of target qubit
    '''
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


def compare(tab, ndx, ctrl, targ):
    '''
    Function that takes in a pair and identifies what Paulis are in it

    tab: Cirq_tableau holding Pauli word
    ndx: int holding index (row) of the Pauli string currently being reduced
    ctrl: int holding index (column) of control qubit
    targ: int holding index (column) of target qubit
    '''

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


def pair_solve(tableau, ndx, ndx_list):
    '''
    Heuristic function to produce a single reduction from one node of a tree to a child

    tableau: Cirq_Tableau holding Pauli word
    ndx: int holding index (row) of current Pauli string being reduced
    '''
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
    tab, single_q, new_ndx_list = prune(tab, ndx_list)
    reduction += single_q # appends to reduction
    
    return tab, reduction, new_ndx_list