import numpy as np
import math




# class that implements tableau and necessary operations
# References -Improved Simulation of Stabilizer Circuits by Scott Aaronson & Daniel Gottesman
# - Google Tableau implementation
class Cirq_Tableau:
    
    def __init__(
        self,
        pauli_word: list[str] = None  
    ):
        if pauli_word is None or len(pauli_word) == 0:
            self._column_num = None
            self._row_num = None

            self._ss, self._xs, self._zs = None, None, None
        else:
            if "-" in pauli_word[0]:
                self._column_num = len(pauli_word[0][1:]) 
            else:
                self._column_num = len(pauli_word[0])
            self._row_num = len(pauli_word)

            self._ss = self.create_sign(pauli_word)
            self._xs, self._zs = self.create_tab(pauli_word)

    # setters & getters 
    @property
    def ss(self) -> np.ndarray:
        return self._ss

    @ss.setter 
    def ss(self, new_ss: np.ndarray):
        self._ss = new_ss
        
    @property
    def xs(self) -> np.ndarray:
        return self._xs

    @xs.setter 
    def xs(self, new_xs: np.ndarray):
        self._xs = new_xs
        
    @property
    def zs(self) -> np.ndarray:
        return self._zs

    @zs.setter 
    def zs(self, new_zs: np.ndarray):
        self._zs = new_zs
        
    @property
    def column_num(self) -> int:
        return self._column_num

    @column_num.setter 
    def column_num(self, new_num: int):
        self._column_num = new_num
    
    @property
    def row_num(self) -> int:
        return self._row_num

    @row_num.setter 
    def row_num(self, new_num: int):
        self._row_num = new_num

    # functions that create parts of tableau
    def create_sign(self, pauli_word: list[str]):
        temp_ss = np.zeros((self.row_num), dtype=int)
        for pauli_string in pauli_word:
            if "-" in pauli_string:
                index = pauli_word.index(pauli_string)
                pauli_word[index] = pauli_string.replace("-", "")
                #print(pauli_word)
                temp_ss[index] = 1
        return temp_ss
        
    def create_tab(self, pauli_word: list[str]):
        temp_x = np.zeros((self.row_num, self.column_num), dtype=int)
        temp_z = np.zeros((self.row_num, self.column_num), dtype=int)
        for i, pauli_string in enumerate(pauli_word):
            for j, pauli in enumerate(pauli_string):
                if pauli == "X" or pauli == "Y":
                    temp_x[i][j] = 1
                if pauli == "Z" or pauli == "Y":
                    temp_z[i][j] = 1
        return temp_x, temp_z

    # Clifford operations on tableau. 
    def apply_H(self,column: int):
        self.ss ^= self.xs[:, column] & self.zs[:, column]
        self.xs[:, column], self.zs[:, column] = self.zs[:, column].copy(), self.xs[:, column].copy()

    def apply_S(self, column: int):
        self.ss ^= self.xs[:, column] & self.zs[:, column]
        self.zs[:, column] = self.xs[:, column] ^ self.zs[:, column]

    def apply_CX(self, control: int, target: int):
        self.ss ^= (
            (self.xs[:, control] & self.zs[:, target])
            &(~(self.xs[:, target] ^ self.zs[:, control]))
        )
        self.xs[:, target] ^= self.xs[:, control]
        self.zs[:, control] ^= self.zs[:, target]

    # class operations necessary for comparisons and equating
    def copy(self):
        new_tab = Cirq_Tableau()
        new_tab.column_num = self.column_num
        new_tab.row_num = self.row_num
        new_tab.ss = self.ss.copy()
        new_tab.zs = self.zs.copy()
        new_tab.xs = self.xs.copy()
        return new_tab
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented  
        return (
            self.column_num == other.column_num
            and self.row_num == other.row_num
            and np.array_equal(self.ss, other.ss)
            and np.array_equal(self.xs, other.xs)
            and np.array_equal(self.zs, other.zs)
        )
    
    def return_string(self):
        string = ''
        for i in range(self.row_num):
            if self.ss[i]:
                    string += "-"  

            for j in range(self.column_num):
                if self.xs[i][j] and not self.zs[i][j]:
                    string += "X"
                elif not self.xs[i][j] and self.zs[i][j]:
                    string += "Z"
                elif self.xs[i][j] and self.zs[i][j]:
                    string += "Y"
                else:
                    string += "I"
            if i < self.row_num - 1:
                # if self.ss[i]:
                #     string = "-" + string[::-1]
                # else:
                #     string = string[::-1]
                string += "\n" 

        return string
    
    def __copy__(self):
        return self.copy()
        

    def __str__(self) -> str:
        ss = np.expand_dims(self.ss, axis = 1)
        xz = np.concatenate((self.xs, self.zs, ss), axis=1)
        return str(xz)

    def __hash__(self) -> int:
        return hash(self.zs.tobytes() + self.xs.tobytes() + self.ss.tobytes())

    def __lt__(self, other):
        return self.row_num < other.row_num



# class to hold node information for tree 
class Node:

    def __init__(
        self,
        state
    ):
        self._ni = 0 # number of simulations 
        self._qi = 0 # cumulative CXs for root to terminal node across simulations
        self._parent = None # node that had action applied to get to current node
        self._action = [] # action applied to parent node to get to current node
        self._action_time = 0 # time it takes to produce the action that led to the state of the Node 
        self._untouched = list(range(state.row_num)) # set of unused actions - actually set of all possible child nodes
        self._dag = None # dag for order preserving work
        self._ndx_list = []
        self._children = [] # nodes produced when actions are applied to current node
        self._state = state # Pauli word representing state of node
        
    # properties and setters 
    @property
    def ni(self) -> int:
        return self._ni

    @ni.setter
    def ni(self, new_ni: int):
        self._ni = new_ni

    @property
    def qi(self) -> int:
        return self._qi

    @qi.setter
    def qi(self, new_num: int):
        self._qi = new_num
        
    @property
    def curr_cx_num(self) -> int:
        return self._curr_cx_num

    @curr_cx_num.setter
    def curr_cx_num(self, new_num: int):
        self._curr_cx_num = new_num
        
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_word):
        self._state = new_word

        
    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        self._parent = new_parent

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, new_action):
        self._action = new_action

    @property
    def action_time(self):
        return self._action_time

    @action_time.setter
    def action_time(self, new_time):
        self._action_time = new_time
    
    @property
    def untouched(self):
        return self._untouched

    @untouched.setter
    def untouched(self, new_list):
        self._untouched = new_list

    @property
    def dag(self):
        return self._dag

    @dag.setter
    def dag(self, new_dag):
        self._dag = new_dag
        
    @property
    def ndx_list(self):
        return self._ndx_list

    @ndx_list.setter
    def ndx_list(self, new_list):
        self._ndx_list = new_list  
        
    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, new_children):
        self._children = new_children

   
    # standard necessary functions    
    def __hash__(self) -> int:
        return hash(self.parent) + hash(str(self.action))

    # useful specific fucntions
    def UCT(self, param):
        
        # if it has been completely unexplored 
        if self.ni == 0:
            return float('inf')

        # if it is a root (may never be used but just in case)
        if self.parent == None:
            # negative cx_num leads to larger values for smaller qi
            return -(self.qi/ self.ni)


        return -(self.qi/self.ni) + (param * math.sqrt(math.log(self.parent.ni)/ self.ni)) 