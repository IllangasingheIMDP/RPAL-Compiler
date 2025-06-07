
from collections import defaultdict
import functools



def apply_unary(cse_machine, rator, unop):
    

    
    rator_value = rator.value

    
    unary_operators = {
            "Print"       : lambda cse_machine, operand: apply_print(cse_machine, operand),
            "Isstring"    : lambda cse_machine, operand: operand.type == "STR",
            "Isinteger"   : lambda cse_machine, operand: operand.type == "INT" ,
            "Istruthvalue": lambda cse_machine, operand: operand.type == "bool",
            "Isfunction"  : lambda cse_machine, operand: operand.type == "lambda",
            "Null"        : lambda cse_machine, operand: operand.type == "nil",
            "Istuple"     : lambda cse_machine, operand: isinstance(operand.value, list) or operand.type == "nil",
            "Order"       : lambda cse_machine, operand: apply_order(cse_machine, operand),
            "Stern"       : lambda cse_machine, operand: apply_stern(cse_machine, operand.value),
            "Stem"        : lambda cse_machine, operand: apply_stem(cse_machine, operand.value),
            "ItoS"        : lambda cse_machine, operand: str(operand.value) if isinstance(operand.value, int) and not isinstance(operand.value, bool) else cse_machine._error_handler.handle_error("CSE : Invalid unary operation"),
            "neg"         : lambda cse_machine, operand: -operand.value if isinstance(operand.value, int) else cse_machine._error_handler.handle_error("CSE : Invalid unary operation"),
            "not"         : lambda cse_machine, operand: not operand.value if isinstance(operand.value, bool) else cse_machine._error_handler.handle_error("CSE : Invalid unary operation"),
        }
    
    
    operation_function = unary_operators.get(unop)

    if operation_function:
        
        return operation_function(cse_machine, rator)
    else:
        
        raise ValueError("Invalid binary operation: " + unop)


def apply_print(cse_machine, operand):

    
    element = operand.value
    
    
    def covert_to_string(element):
        if isinstance(element, list):
            out = ""
            return convert_list(element,out)
        elif element == "lambda":
            x = "".join(x for x in operand.bounded_variable)
            k = str(operand.control_structure)
            return "[lambda closure: " + x + ": " + k + "]"
        elif isinstance(element, bool):
            return "true" if element else "false"
        elif isinstance(element, str):
            return element
        elif isinstance(element, int):
            return str(element)
        elif element == None:
            return "nil"
        else:
            raise TypeError("Unknown element type.")
        
    def convert_list(element,out):
        if isinstance(element, list):
            out += "("
            for el in element:
                out = convert_list(el,out)
            out = out[:-2] +  ")"
        else:
            if isinstance(element.value, list):
                out += "("
                for el in element.value:
                    out = convert_list(el,out)
                out = out[:-2] +  "), "
            else:
                out += covert_to_string(element.value) + ", "
        return out
    
    
    cse_machine._print_queue.append(covert_to_string(element).replace("\\n", "\n").replace("\\t", "\t"))
    
    
    return "dummy"



def apply_order(cse_machine, operand):

    
    if isinstance(operand.value, list):
        return len(operand.value)
    elif operand.type == "nil":
        return 0
    else:
        cse_machine._error_handler.handle_error("CSE : Invalid unary operation")


def apply_stern(cse_machine, operand):

    
    if isinstance(operand, str) and len(operand) >= 1:
        return operand[1:]
    else:
        cse_machine._error_handler.handle_error("CSE : Invalid unary operation")


def apply_stem(cse_machine, operand):

    if isinstance(operand, str) and len(operand) >= 1:
        return operand[0]
    else:
        cse_machine._error_handler.handle_error("CSE : Invalid unary operation")
        

def apply_binary(cse_machine, rator, rand, binop):
   

    
    binary_operators = {
            "aug" : lambda cse_machine, rator, rand : apply_aug(cse_machine, rator, rand),
            "or"  : lambda cse_machine, rator, rand : apply_or(cse_machine, rator, rand),
            "&"   : lambda cse_machine, rator, rand : apply_and(cse_machine, rator, rand),
            "+"   : lambda cse_machine, rator, rand : apply_arithmetic(cse_machine, rator, rand, lambda a, b: a + b),
            "-"   : lambda cse_machine, rator, rand : apply_arithmetic(cse_machine, rator, rand, lambda a, b: a - b),
            "*"   : lambda cse_machine, rator, rand : apply_arithmetic(cse_machine, rator, rand, lambda a, b: a * b),
            "/"   : lambda cse_machine, rator, rand : apply_arithmetic(cse_machine, rator, rand, lambda a, b: a // b),
            "**"  : lambda cse_machine, rator, rand : apply_arithmetic(cse_machine, rator, rand, lambda a, b: a ** b),
            "gr"  : lambda cse_machine, rator, rand : apply_comparison(cse_machine, rator, rand, lambda a, b: a > b),
            "ge"  : lambda cse_machine, rator, rand : apply_comparison(cse_machine, rator, rand, lambda a, b: a >= b),
            "ls"  : lambda cse_machine, rator, rand : apply_comparison(cse_machine, rator, rand, lambda a, b: a < b),
            "le"  : lambda cse_machine, rator, rand : apply_comparison(cse_machine, rator, rand, lambda a, b: a <= b),
            "eq"  : lambda cse_machine, rator, rand : apply_eq(cse_machine, rator, rand),
            "ne"  : lambda cse_machine, rator, rand : apply_ne(cse_machine, rator, rand),
            "Conc": lambda cse_machine, rator, rand : apply_conc(cse_machine, rator, rand)
                    }

    
    operation_function = binary_operators.get(binop)
    if operation_function:
        
        return operation_function(cse_machine, rator, rand)
    else:
        
        raise ValueError("Invalid binary operation: " + binop)


def apply_aug(cse_machine, rator, rand):
    
    
    if rator.type == "nil" :
        return ControlStructureElement("tuple", [rand])
    elif rand.type == "nil":
        if rator.type == "tuple":
            
            rator.value.append(rand)
            return rator
        return rator
    elif rator.type in ["tuple","ID","INT","STR","bool"] and rand.type in ["tuple","ID","INT","STR","bool"]:
        if isinstance(rator.value, list) :
            ls = rator.value.copy()
            ls.append(rand)
            res = ControlStructureElement("tuple", ls)
            return res
        return ControlStructureElement("tuple", [rator, rand])
    else:
        return cse_machine._error_handling.handle_error("Cannot augment a non tuple (2).")


def apply_or(cse_machine, rator, rand):
    
    if isinstance(rator, bool) and isinstance(rand, bool):
        
        return rator or rand
    else:
        
        raise cse_machine._error_handling.handle_error("Invalid value used in logical expression 'or'")


def apply_and(cse_machine, rator, rand):
    
    if isinstance(rator, bool) and isinstance(rand, bool):
        
        return rator and rand
    else:
        
        raise ValueError("Illegal Operands for '&'")


def apply_eq(cse_machine, rator, rand):
    
    if type(rator) == type(rand):
        
        return rator == rand
    else:
        
        raise cse_machine._error_handling.handle_error("Illegal Operands for 'eq'")


def apply_ne(cse_machine, rator, rand):
    
    if type(rator) == type(rand):
        
        return rator != rand
    else:
        
        raise cse_machine._error_handling.handle_error("Illegal Operands for 'ne'")


def apply_arithmetic(cse_machine, rator, rand, operation):
    
    if isinstance(rator, int) and isinstance(rand, int):
        
        return operation(rator, rand)
    else:
        
        raise cse_machine._error_handler.handle_error("Illegal Operands for Arithmetic Operation")

def apply_conc(cse_machine,rator,rand):
    
    
    if isinstance(rator, str) and isinstance(rand, str):
        return rator + rand
    else:
        raise cse_machine._error_handler.handle_error("Non-strings used in conc call")
    

def apply_comparison(cse_machine, rator, rand, operation):
    
    if (isinstance(rator, int) and isinstance(rand, int)) or (isinstance(rator, str) and isinstance(rand, str)):
        
        return operation(rator, rand)
    else:
        
        raise cse_machine._error_handler.handle_error("Illegal Operands for 'gr'")


def var_lookup(cse_machine , var_name):
    env_pointer = cse_machine.current_enviroment
    while env_pointer:
        if var_name in env_pointer._environment:
            out = env_pointer._environment[var_name]
            return out
        env_pointer = env_pointer.parent
    else:
        cse_machine._error_handler.handle_error(f"CSE : Variable [{var_name}] not found in the environment")
def add_table_data(cse_machine,rule):
    table_data = cse_machine.table_data
    table_data.append((rule,cse_machine.control.whole_stack()[:],cse_machine.stack.whole_stack()[:],[cse_machine.current_enviroment.index][:]))

def add_table_data_decorator(table_entry):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self._add_table_data(table_entry)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def convert_list(element,out):
 
    if isinstance(element, list):
        out += "("
        for el in element:
            out = convert_list(el,out)
        out = out[:-1] +  ")"
    else:
        if isinstance(element.value, list):
            out += "("
            for el in element.value:
                out = convert_list(el,out)
            out = out[:-1] +  "),"
        else:
            out += str(element.value) + ","
    return out

def raw(string):
    return string.encode('unicode_escape').decode()

class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)
    
    def whole_stack(self):
        return self.items
class Environment:
    index = -1

    
    INITIAL_VARIABLES = [
                            "Print", "Isstring", "Isinteger", "Istruthvalue",
                            "Istuple", "Isfunction", "Null", "Order", "Stern",
                            "Stem", "ItoS", "neg", "not", "Conc"
                         ]

    def __init__(self, parent=None):
        Environment.index += 1
        self.index = Environment.index
        self._environment = defaultdict(lambda: [None, None])  
        self.parent = parent

        
        if self.index == 0:
            self._initialize_initial_vars()

    def _initialize_initial_vars(self):
        initial_vars = {var: ["inbuilt-functions",None] for var in self.INITIAL_VARIABLES}
        self._environment.update(initial_vars)

    def add_var(self, name, type, value):
        self._environment[name] = [type, value]

    def add_child(self, branch):
        self.children.append(branch)

    def set_parent(self, parent):

        self.parent = parent
        
        self._environment['__parent__'] = parent._environment if parent else None

    def reset_index(self):

        Environment.index = -1

class ControlStructure(Stack):
    def __init__(self, index):
        super().__init__()
        self.elements = self.items
        self.index = index

class ControlStructureElement:
    def __init__(self, type, value, bounded_variable=None,control_structure=None, env=None , operator=None):
        self.type = type
        self.value = value
        self.bounded_variable = bounded_variable
        self.control_structure = control_structure
        self.env = env
        self.operator = operator
class Linearizer:

    def __init__(self):

        self.control_structures = []
        
    def linearize(self,st_tree):
        self.preorder_traversal(st_tree, 0)
        
        return self.control_structures
    
    def preorder_traversal(self, root , index):
        
        if len(self.control_structures) <= index:
            self.control_structures.append(ControlStructure(index))
            
        if not root.children:	
            self.control_structures[index].push(ControlStructureElement(self.filter(root.data)[0], self.filter(root.data)[1]))
            return
        
        if root.data == "lambda":
            
            if root.children[0].data == ",": 
                var_list = []
                for child in root.children[0].children:
                    var_list.append(self.filter(child.data)[1])
                self.control_structures[index].push(ControlStructureElement("lambda", "lambda", var_list, len(self.control_structures)))
            else:
                self.control_structures[index].push(ControlStructureElement("lambda", "lambda", [self.filter(root.children[0].data)[1]], len(self.control_structures)))
            self.preorder_traversal(root.children[1], len(self.control_structures))
            
        elif root.data == "tau":
            self.control_structures[index].push(ControlStructureElement("tau", len(root.children)))
            for child in root.children:
                self.preorder_traversal(child, index)

        elif root.data == "->":
            self.control_structures[index].push(ControlStructureElement("delta", "delta",None, len(self.control_structures)))
            self.preorder_traversal(root.children[1], len(self.control_structures))
            self.control_structures[index].push(ControlStructureElement("delta", "delta",None, len(self.control_structures)))
            self.preorder_traversal(root.children[2], len(self.control_structures))
            self.control_structures[index].push(ControlStructureElement("beta", "beta"))
            self.preorder_traversal(root.children[0], index)
        
        else:
            self.control_structures[index].push(ControlStructureElement(self.filter(root.data)[0], self.filter(root.data)[1]))
                
            self.preorder_traversal(root.children[0], index)
            if len(root.children) > 1:
                self.preorder_traversal(root.children[1], index)
    
    def filter(self,token):
        output = list()
        if token[0] == "<":
                if len(token)>3 and token[1:3] == "ID":
                    if token[4:-1] in ["Conc","Print","Stern","Stem","Isstring","Isinteger","Istruthvalue","Isfunction","Null","Istuple","Order","ItoS","not","neg"	]:
                        output = [token[4:-1], token[4:-1]]
                    else:
                        output = ["ID", token[4:-1]]
                elif len(token)>4 and token[1:4] == "INT":
                    output = ["INT", int(token[5:-1])]
                elif len(token)>4 and token[1:4] == "STR":
                    output = ["STR", token[6:-2]]
                elif token[1:-1] == "true":
                    output = ["bool",True]
                elif token[1:-1] == "false":
                    output = ["bool",False]
                elif token[1:-1] == "nil":
                    output = ["nil",None]
                else:
                    output = [token[1:-1], token[1:-1]]
        else:
            output = [token, token]
        return output
    

class CseErrorHandler:

    def __init__(self, cse_machine):

        self.cse_machine = cse_machine

    def handle_error(self,message):

       
        raise Exception(message)


class CSEMachine:


    def __init__(self):

        self._error_handler = CseErrorHandler(self)

        
        self._linearizer = Linearizer()
        
        
        
        self.primitive_environment = Environment()

        
        self.control_structures = None
        self.current_enviroment = self.primitive_environment
        self.stack = Stack()
        self.control = Stack()
        
        
        self._print_queue = list()
        self.table_data = list()

        

        self.binary_operator = {
                                
                                "+", "-", "/", "*", "**", 
                                
                                "eq", "ne", "gr", "ge", "le","ls",
                                
                                ">", "<", ">=", "<=", 
                                
                                "or", "&", "aug",  
                                
                                "Conc"
                                }
        
        

        self.unary_operators =  {
                                
                                "neg", "not",
                                
                                "Print", 
                                
                                "Isstring", "Isinteger", "Istruthvalue", "Isfunction", "Null","Istuple",
                                
                                "Order", "Stern", "Stem", "ItoS", "$ConcPartial"
                                }
        
        

        self.inbuilt_functions = {
                                
                                "Print", 
                                
                                "Isstring", "Isinteger", "Istruthvalue", "Isfunction", "Null","Istuple",
                                
                                "Order", "Stern", "Stem", "ItoS", "$ConcPartial"
                                }
    def initialize(self):

    
        
        primitive_enviroment = ControlStructureElement("env_marker", "env_marker", None, None, self.current_enviroment)

        
        self.stack.push(primitive_enviroment)
        self.control.push(primitive_enviroment)

        
        if self.control_structures:
            elements = self.control_structures[0].elements
            for element in elements:
                self.control.push(element)
        else:
            
            self._error_handler.handle_error("Control structures are empty")

    def execute(self, st_tree):

        
        
        self.control_structures = self._linearizer.linearize(st_tree)
        
        self.initialize()
        
        
        while not self.control.is_empty():

            
            control_top = self.control.peek()
            
            stack_top = self.stack.peek()

            

            if control_top.type in ['ID','STR','INT','bool','tuple','Y*','nil','dummy']:
                self.CSErule1()
            elif control_top.type == "lambda":
                self.CSErule2()
            elif control_top.type == "env_marker":
                self.CSErule5()
            elif control_top.value in self.binary_operator and self.stack.size() >= 2:
                self.CSErule6()
            elif control_top.value in self.unary_operators and self.stack.size() >= 1:
                self.CSErule7()
            elif control_top.type == "beta" and self.stack.size() >= 1:
                self.CSErule8()
            elif control_top.type == "tau":
                self.CSErule9()
            elif control_top.type == "gamma" and stack_top.type == "tuple":
                self.CSErule10()
            elif control_top.type == "gamma" and stack_top.type == "Y*":
                self.CSErule12()
            elif control_top.type == "gamma" and stack_top.type == "eta":
                self.CSErule13()
            elif control_top.type == "gamma"  and stack_top.type == "lambda":
                    if len(stack_top.bounded_variable) > 1:
                        self.CSErule11()
                    else:
                        self.CSErule4()
            elif control_top.type == "gamma" and stack_top.type == "ConcPartial":
                self.Concpartial()
            else:
                self._error_handler.handle_error("CSE : Invalid control structure")

    @add_table_data_decorator("1")
    def CSErule1(self):

        control_top = self.control.peek()
        if control_top.type in ['STR','INT','bool','tuple','Y*','nil','dummy']:
            self.stack.push(self.control.pop())
        else :
            item = self.control.pop()
            var_name = item.value
            var = self._var_lookup(var_name)
            if var[0] == "eta" or var[0] == "lambda":
                self.stack.push(var[1])
            else :
                self.stack.push(ControlStructureElement(var[0],var[1]))
        
    @add_table_data_decorator("2") 
    def CSErule2(self):
 
        lambda_ = self.control.pop()
        lambda_.env = self.current_enviroment
        self.stack.push(lambda_)
        
    @add_table_data_decorator("3")
    def CSErule3(self):
        pass
    
    @add_table_data_decorator("4")
    def CSErule4(self):


        
        if self.current_enviroment.index >= 10000:
            self._error_handler.handle_error("CSE : Environment limit exceeded")
            return

        self.control.pop()
        lambda_ = self.stack.pop()
        rand = self.stack.pop()
        new_enviroment = Environment()
        if rand.type  == "eta" or rand.type == "lambda":
            new_enviroment.add_var(lambda_.bounded_variable[0],rand.type,rand)
        elif rand.type in ["tuple","INT","bool","STR","nil"]:
            new_enviroment.add_var(lambda_.bounded_variable[0],rand.type,rand.value)
        else:
            self._error_handler.handle_error("CSE : Invalid type")
        new_enviroment.parent = lambda_.env

        self.current_enviroment = new_enviroment
        
        new_enviroment_element = ControlStructureElement("env_marker","env_marker",None,None,new_enviroment)
        
        self.control.push(new_enviroment_element) 
        
        for element in self.control_structures[lambda_.control_structure].elements:
            self.control.push(element)
            
        self.stack.push(new_enviroment_element)
        
    @add_table_data_decorator("5")
    def CSErule5(self):

        env = self.control.pop().env
        value = self.stack.pop()
        if env == self.stack.pop().env:
            self.stack.push(value)
            for element in reversed(self.stack.whole_stack()):
                if element.type == "env_marker":
                    self.current_enviroment = element.env
                    break
        else:
            self._error_handler.handle_error("CSE : Invalid environment")
                
    @add_table_data_decorator("6")
    def CSErule6(self):

        binop = self.control.pop().value
        rator = self.stack.pop()
        rand = self.stack.pop()
        if binop == "aug":
            self.stack.push(self._apply_binary(rator,rand,binop))
        elif binop == "Conc":
            if rator.type == "STR" and rand.type == "STR":
                result =self._apply_binary(rator.value,rand.value,binop)
                self.stack.push(ControlStructureElement("STR",result))
                self.remove_gamma()
                self.remove_gamma()
            elif rator.type == "STR":
                rator.type = "ConcPartial"
                self.stack.push(rand)
                self.stack.push(rator)
                self.remove_gamma()
            else:
                self._error_handler.handle_error("CSE : Invalid type for concatenation")
        else:
            rator = rator.value
            rand = rand.value
            result = self._apply_binary(rator,rand,binop)
            typ  = None
            if type(result) == bool:
                typ = "bool"
            else:
                typ = "INT"
            self.stack.push(ControlStructureElement(typ,result))
        
    @add_table_data_decorator("7")    
    def CSErule7(self):

        unop = self.control.pop().value
        rator_e = self.stack.pop()
        result = self._apply_unary(rator_e,unop)
        res_type = None
        if type(result) == bool:
            res_type = "bool"
        elif type(result) == str:
            res_type = "STR"
        else :
            res_type = "INT"
        if unop in self.inbuilt_functions:
            self.remove_gamma()
        self.stack.push(ControlStructureElement(res_type,result))
                    
    @add_table_data_decorator("8")
    def CSErule8(self):

        val = self.stack.pop().value
        if val == True :
            self.control.pop()
            self.control.pop()
            delta = self.control.pop()
            for element in self.control_structures[delta.control_structure].elements:
                self.control.push(element)
        elif val == False:
            self.control.pop()
            delta = self.control.pop()
            self.control.pop()
            for element in self.control_structures[delta.control_structure].elements:
                self.control.push(element)
        else:
            self._error_handler.handle_error("CSE : Invalid type for condition")

    @add_table_data_decorator("9")
    def CSErule9(self):

        tau = self.control.pop()
        n = tau.value
        tup = []
        for i in range(n):
            tup.append(self.stack.pop())
        self.stack.push(ControlStructureElement("tuple",tup))

    @add_table_data_decorator("10")   
    def CSErule10(self):

        self.control.pop()
        l = self.stack.pop()
        index = self.stack.pop()
        if index.type != "INT":
            self._error_handler.handle_error("CSE : Invalid index")
        index  = index.value-1
        self.stack.push(l.value[index])
                
    @add_table_data_decorator("11")
    def CSErule11(self):

        self.control.pop()
        lambda_ = self.stack.pop()
        var_list = lambda_.bounded_variable
        k = lambda_.control_structure
        c = lambda_.env
        
        new_env = Environment()
        rand = self.stack.pop()
        
        if len(var_list) != len(rand.value):
            self._error_handler.handle_error("CSE : Invalid number of arguments")
            
        for i in range(len(var_list)):
            if rand.value[i].type == "eta" or rand.value[i].type == "lambda":
                new_env.add_var(var_list[i],rand.value[i].type,rand.value[i])
            else:
                new_env.add_var(var_list[i],rand.value[i].type,rand.value[i].value)
        
        new_env.parent = c
        self.current_enviroment = new_env
        env_marker = ControlStructureElement("env_marker","env_marker",None,None,new_env)
        self.stack.push(env_marker)
        self.control.push(env_marker)
        
        for element in self.control_structures[k].elements:
            self.control.push(element)

    @add_table_data_decorator("12")       
    def CSErule12(self):

        self.control.pop()
        self.stack.pop()
        lambda_ = self.stack.pop()
        if lambda_.type != "lambda":
            self._error_handler.handle_error("CSE : expected lambda")
        eta = ControlStructureElement("eta","eta",lambda_.bounded_variable,lambda_.control_structure,lambda_.env)
        self.stack.push(eta)
        
    @add_table_data_decorator("13")
    def CSErule13(self):

        self.control.push(ControlStructureElement("gamma","gamma"))
        eta = self.stack.peek()
        self.stack.push(ControlStructureElement("lambda","lambda",eta.bounded_variable,eta.control_structure,eta.env))
    
    def Concpartial(self):
        rator = self.stack.pop()
        rand = self.stack.pop()
        if rand.type == "STR":
            result = self._apply_binary(rator.value,rand.value,"Conc")
            self.stack.push(ControlStructureElement("STR",result))
            self.remove_gamma()
        else:
            self._error_handler.handle_error("CSE : Invalid type for concatenation")

            
    
    
    
    
    def _var_lookup(self , var_name):
        return var_lookup(self, var_name)
            
    def _apply_binary(self , rator , rand , binop):
        return apply_binary(self, rator, rand, binop)
                
    def _apply_unary(self , rator , unop):
        return apply_unary(self, rator, unop)     

    def _add_table_data(self, rule):
        add_table_data(self, rule)

    
    def _generate_output(self):
        return "".join(self._print_queue)+"\n"
    
    def _generate_raw_output(self):
        return raw(self._generate_output()) 
    
    def remove_gamma(self):
        if self.control.peek().type == "gamma":
            self.control.pop()    
    


