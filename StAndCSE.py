import sys
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from parser import ASTNode, parse_rpal
from lexical import rpal_lexer
from Standardizer import STNode, ast_to_st



# --- CSE Machine Values ---
class CSEValue:
    """Base class for CSE values"""
    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return self.__str__()

class CSEInteger(CSEValue):
    def __init__(self, value: int):
        self.value = value
    def __str__(self): return str(self.value)

class CSEString(CSEValue):
    def __init__(self, value: str):
        self.value = value
    def __str__(self): return f'"{self.value}"'

class CSEBoolean(CSEValue):
    def __init__(self, value: bool):
        self.value = value
    def __str__(self): return str(self.value).lower()

class CSENil(CSEValue):
    def __str__(self): return "nil"

class CSEDummy(CSEValue):
    def __str__(self): return "dummy"

class CSETuple(CSEValue):
    def __init__(self, elements: List[CSEValue]):
        self.elements = elements
    def __str__(self): return " ".join(map(str, self.elements))
    def __len__(self): return len(self.elements)
    def __getitem__(self, index): return self.elements[index]

class CSEEnvironment:
    _next_id = 0
    def __init__(self, parent: Optional["CSEEnvironment"] = None):
        self.bindings: Dict[str, CSEValue] = {}
        self.parent = parent
        self.id = CSEEnvironment._next_id
        CSEEnvironment._next_id += 1
    
    def lookup(self, name: str) -> CSEValue:
        
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent:
            return self.parent.lookup(name)
        else:
            raise CSERuntimeError(f"Name '{name}' is not defined")
    
    def bind(self, name: str, value: CSEValue):
        self.bindings[name] = value
        
    def __str__(self):
        items = ', '.join(f"{k}={v}" for k, v in self.bindings.items())
        parent_id = f" parent={self.parent.id}" if self.parent else ""
        return f"Env{self.id}({items}{parent_id})"
    def __repr__(self): return self.__str__()

class CSEClosure(CSEValue):
    def __init__(self, var_name: str, body: STNode, env: CSEEnvironment):
        self.var_name = var_name
        self.body = body
        self.env = env
    def __str__(self): return f"Closure({self.var_name} -> {self.body} in Env{self.env.id})"

class CSEBuiltinFunction(CSEValue):
    def __init__(self, name: str):
        self.name = name
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        raise NotImplementedError
    def __str__(self): return f"Builtin:{self.name}"

class CSEStem(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("stem")
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if isinstance(arg, CSEString):
            machine.stack.append(CSEString(arg.value[1:] if len(arg.value) > 0 else ""))
        else:
            raise CSERuntimeError(f"Stem requires string argument, got {arg}")
class CSEIsInteger(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("Isinteger")
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if isinstance(arg, CSEInteger):
            machine.stack.append(CSEBoolean(True))
        else:
            machine.stack.append(CSEBoolean(False))

class CSEStern(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("stern")
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if isinstance(arg, CSEString):
            machine.stack.append(CSEString(arg.value[1:] if len(arg.value) > 0 else ""))
        else:
            raise CSERuntimeError(f"Stern requires string argument, got {arg}")

class CSEYCombinator(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("Y")
    
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if not isinstance(arg, CSEClosure):
            raise CSERuntimeError(f"Y combinator requires a function argument, got {arg}")
        
        # Create a recursive closure
        rec_closure = CSERecursiveClosure(arg)
        machine.stack.append(rec_closure)

class CSERecursiveClosure(CSEClosure):
    def __init__(self, eta_closure: CSEClosure):
        super().__init__(eta_closure.var_name, eta_closure.body, eta_closure.env)
        self.eta_closure = eta_closure
    
    def __str__(self): return f"RecClosure({self.eta_closure})"

class CSEPrint(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("Print")
    
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if isinstance(arg, CSETuple):
            # Print each element of the tuple
            print(arg)
        else:
            # Print a single value
            print(arg)
        
        # Return dummy
        machine.stack.append(CSEDummy())

class CSEOrder(CSEBuiltinFunction):
    def __init__(self):
        super().__init__("Order")
    
    def apply(self, machine: "CSEMachine", arg: CSEValue):
        if isinstance(arg, CSETuple):
            # Return the length of the tuple
            machine.stack.append(CSEInteger(len(arg.elements)))
        else:
            raise CSERuntimeError(f"Order requires a tuple argument, got {arg}")

# --- CSE Machine ---
class CSEMachine:
    """
    Control Structure Evaluator (CSE) Machine for RPAL
    """
    def __init__(self):
        self.control = []
        self.stack = []
        self.env = CSEEnvironment()
        self.env_stack = []
        self.debug = False  # Disable debug output
        
        # Initialize built-in functions
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in functions"""
        self.env.bind("Y", CSEYCombinator())
        self.env.bind("Print", CSEPrint())
        self.env.bind("Order", CSEOrder())
        self.env.bind("stem", CSEStem())
        self.env.bind("stern", CSEStern())
        self.env.bind("Isinteger", CSEIsInteger())
        # Add other built-ins as needed
    
    def evaluate(self, st_node: STNode, ast_only=False) -> CSEValue:
        """Evaluate a standardized tree"""
        if ast_only:
            return None
            
        self.control = [st_node]
        self.stack = []
        self.env_stack = []
        
        step = 0
        while self.control:
            item = self.control.pop()
            if isinstance(item, STNode):
                self._process_node(item)
            elif isinstance(item, CSEOperation):
                item.apply(self)
            else:
                raise CSERuntimeError(f"Invalid control item: {item}")
            step += 1
        
        if not self.stack:
            return CSENil()
        return self.stack[0]
    
    def _process_node(self, node: STNode):
        node_type = node.node_type

        if node_type == "lambda":
            var_name = node.children[0].value
            body = node.children[1]
            closure = CSEClosure(var_name, body, self.env)
            self.stack.append(closure)
            
        elif node_type == "gamma":
            self.control.append(CSEApply())
            self.control.append(node.children[0])  # Function
            self.control.append(node.children[1])  # Argument
        elif node_type == "@":
            self.control.append(CSEAtOperation())
            self.control.append(node.children[2])  # Right operand
            self.control.append(node.children[1])  # Function
            self.control.append(node.children[0])  # Left operand
        elif node_type == "tuple_apply":
            self.control.append(CSETupleApply(node.children[2]))  # Unpacking info
            self.control.append(node.children[0])  # Function (lambda chain)
            self.control.append(node.children[1])  # Tuple expression

        elif node_type == "Ystar":
            self.stack.append(node)

        elif node_type == "identifier":
            value = self.env.lookup(node.value)
            self.stack.append(value)
            
        elif node_type == "integer":
            self.stack.append(CSEInteger(node.value))
            
        elif node_type == "string":
            self.stack.append(CSEString(node.value))
            
        elif node_type == "true":
            self.stack.append(CSEBoolean(True))
            
        elif node_type == "false":
            self.stack.append(CSEBoolean(False))
            
        elif node_type == "nil":
            self.stack.append(CSENil())
            
        elif node_type == "dummy":
            self.stack.append(CSEDummy())
            
        elif node_type == "tau":
            n = len(node.children)
            self.control.append(CSETupleConstructor(n))
            for child in reversed(node.children):
                self.control.append(child)
                
        elif node_type == "->":
            condition = node.children[0]
            then_branch = node.children[1]
            else_branch = node.children[2]
            self.control.append(CSEConditional(then_branch, else_branch))
            self.control.append(condition)
            
        elif node_type in ["+", "-", "*", "/", "**"]:
            op_map = {
                "+": CSEAdd(),
                "-": CSESubtract(),
                "*": CSEMultiply(),
                "/": CSEDivide(),
                "**": CSEPower()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        elif node_type in ["gr", "ge", "ls", "le", "eq", "ne"]:
            op_map = {
                "gr": CSEGreater(),
                "ge": CSEGreaterEqual(),
                "ls": CSELess(),
                "le": CSELessEqual(),
                "eq": CSEEqual(),
                "ne": CSENotEqual()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        elif node_type in ["or", "&"]:
            op_map = {
                "or": CSEOr(),
                "&": CSEAnd()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        elif node_type in ["neg", "not"]:
            op_map = {
                "neg": CSENeg(),
                "not": CSENot()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            
        elif node_type == "aug":
            self.control.append(CSEAug())
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        else:
            raise CSERuntimeError(f"Unsupported node type: {node_type}")

# --- CSE Operations ---
class CSEOperation:
    """Base class for CSE operations"""
    def apply(self, machine: CSEMachine):
        raise NotImplementedError
    def __str__(self): return self.__class__.__name__

class CSEAtOperation(CSEOperation):
    def apply(self, machine: CSEMachine):
        right = machine.stack.pop()
        fn = machine.stack.pop()
        left = machine.stack.pop()
        
        # Create gamma application: (fn left) right
        gamma_inner = STNode("gamma")
        gamma_inner.add_child(fn)
        gamma_inner.add_child(left)
        
        gamma_outer = STNode("gamma")
        gamma_outer.add_child(gamma_inner)
        gamma_outer.add_child(right)
        
        machine.control.append(gamma_outer)
class CSEApply(CSEOperation):
    """Apply operation for function application"""
    def apply(self, machine: CSEMachine):
        # --- Handle Ystar + Closure pattern for neta closure creation ---
        if (
            len(machine.stack) >= 2 and
            isinstance(machine.stack[-1], STNode) and machine.stack[-1].node_type == "Ystar" and
            isinstance(machine.stack[-2], CSEClosure)
        ):
            if machine.debug:
                print("Detected Ystar and Closure pattern on stack for neta closure creation")
            ystar_node = machine.stack.pop()  # Remove Ystar node
            closure = machine.stack.pop()     # Remove closure

            # Create a neta closure (recursive closure) with same attributes as closure
            neta_closure = CSERecursiveClosure(closure)
            if machine.debug:
                print(f"Created neta closure: {neta_closure}")
            machine.stack.append(neta_closure)
            return  # Do not proceed with normal apply

        # --- Handle neta closure on stack for recursion ---
        if (
            len(machine.stack) >= 1 and
            isinstance(machine.stack[-1], CSERecursiveClosure)
        ):
            neta = machine.stack[-1]
            if machine.debug:
                print("Detected neta closure on stack for recursion handling")
            # Instead of popping CSEApply, append another CSEApply to control (leave the current one)
            machine.control.append(CSEApply())
            machine.control.append(CSEApply())
            # Instead of popping neta, create a lambda closure with same attributes and push to stack
            lambda_closure = CSEClosure(neta.var_name, neta.body, neta.env)
            if machine.debug:
                print(f"Created lambda closure from neta closure: {lambda_closure}")
            machine.stack.append(lambda_closure)
            return  # Do not proceed with normal apply

        # --- Normal function application ---
        fn = machine.stack.pop()
        arg = machine.stack.pop()
        
        if isinstance(fn, CSEClosure):
            # Create a new environment with the closure's environment as parent
            new_env = CSEEnvironment(fn.env)
            
            # Regular binding for non-tuple cases
            new_env.bind(fn.var_name, arg)
            
            # Save the current environment
            old_env = machine.env
            machine.env_stack.append(old_env)
            
            # Set the new environment
            machine.env = new_env
            
            # Push a marker to restore the environment
            machine.control.append(CSERestoreEnv())
            
            # Push the function body
            machine.control.append(fn.body)
            
        elif isinstance(fn, CSEBuiltinFunction):
            # Apply the built-in function
            fn.apply(machine, arg)
            
        elif isinstance(fn, CSERecursiveClosure):
            # Apply the eta closure to the recursive closure itself
            new_env = CSEEnvironment(fn.env)
            new_env.bind(fn.var_name, fn)
            
            # Save the current environment
            old_env = machine.env
            machine.env_stack.append(old_env)
            
            # Set the new environment
            machine.env = new_env
            
            # Push a marker to restore the environment
            machine.control.append(CSERestoreEnv())
            
            # Push the function body
            machine.control.append(fn.body)
            
            # Push the argument back on the stack
            machine.stack.append(arg)
            
        else:
            raise CSERuntimeError(f"Cannot apply non-function: {fn}")
class CSETupleApply(CSEOperation):
    """Special operation for tuple unpacking and application"""
    def __init__(self, unpack_node: STNode):
        self.unpack_node = unpack_node  # Contains variable names to unpack
    
    def apply(self, machine: CSEMachine):
        # Pop function (lambda chain) and tuple
        fn = machine.stack.pop()
        tuple_val = machine.stack.pop()
        
        if not isinstance(tuple_val, CSETuple):
            raise CSERuntimeError(f"Expected tuple for unpacking, got {tuple_val}")
        
        # Get variable names from the unpack node
        var_names = []
        for child in self.unpack_node.children:
            if child.node_type == "identifier":
                var_names.append(child.value)
        
        # Check if the number of variables matches the tuple size
        if len(var_names) != len(tuple_val.elements):
            raise CSERuntimeError(f"Tuple unpacking mismatch: {len(var_names)} variables but {len(tuple_val.elements)} values")
        
        # Create a new environment with the closure's environment as parent
        new_env = CSEEnvironment(fn.env)
        
        # Bind each variable to its corresponding tuple element
        for i, var_name in enumerate(var_names):
            new_env.bind(var_name, tuple_val.elements[i])
        
        # Save the current environment
        old_env = machine.env
        machine.env_stack.append(old_env)
        
        # Set the new environment
        machine.env = new_env
        
        # Push a marker to restore the environment
        machine.control.append(CSERestoreEnv())
        
        # Push the function body (innermost lambda body)
        current_fn = fn
        while isinstance(current_fn, CSEClosure) and len(var_names) > 0:
            var_names.pop()  # Remove one variable name
            if len(var_names) == 0:
                # This is the innermost lambda, push its body
                machine.control.append(current_fn.body)
                break
            else:
                # Get the next lambda in the chain
                if isinstance(current_fn.body, STNode) and current_fn.body.node_type == "lambda":
                    current_fn = CSEClosure(current_fn.body.children[0].value, current_fn.body.children[1], new_env)
                else:
                    raise CSERuntimeError("Invalid lambda chain for tuple unpacking")

class CSERestoreEnv(CSEOperation):
    """Restore the previous environment"""
    def apply(self, machine: CSEMachine):
        if machine.env_stack:
            machine.env = machine.env_stack.pop()
        else:
            raise CSERuntimeError("Environment stack underflow")

class CSETupleConstructor(CSEOperation):
    """Construct a tuple from n elements on the stack"""
    def __init__(self, size: int):
        self.size = size
    
    def apply(self, machine: CSEMachine):
        if len(machine.stack) < self.size:
            raise CSERuntimeError(f"Not enough elements on stack for tuple of size {self.size}")
        
        elements = []
        for _ in range(self.size):
            elements.append(machine.stack.pop())
        
        elements.reverse()  # Reverse to maintain original order
        machine.stack.append(CSETuple(elements))
    
    def __str__(self): return f"TupleConstructor({self.size})"

class CSEConditional(CSEOperation):
    """Conditional operation (if-then-else)"""
    def __init__(self, then_branch: STNode, else_branch: STNode):
        self.then_branch = then_branch
        self.else_branch = else_branch
    
    def apply(self, machine: CSEMachine):
        condition = machine.stack.pop()
        
        if not isinstance(condition, CSEBoolean):
            raise CSERuntimeError(f"Condition must be a boolean, got {condition}")
        
        if condition.value:
            machine.control.append(self.then_branch)
        else:
            machine.control.append(self.else_branch)
    
    def __str__(self): return "Conditional"

# --- Binary Operations ---
class CSEBinaryOp(CSEOperation):
    """Base class for binary operations"""
    def apply(self, machine: CSEMachine):
        
        right = machine.stack.pop()
        left = machine.stack.pop()
        
        result = self._compute(left, right)
        machine.stack.append(result)
    
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        raise NotImplementedError

class CSEAdd(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEInteger(left.value + right.value)
        raise CSERuntimeError(f"Cannot add {left} and {right}")

class CSESubtract(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEInteger(right.value - left.value)
        raise CSERuntimeError(f"Cannot subtract {right} from {left}")

class CSEMultiply(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEInteger(left.value * right.value)
        raise CSERuntimeError(f"Cannot multiply {left} and {right}")

class CSEDivide(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            if right.value == 0:
                raise CSERuntimeError("Division by zero")
            return CSEInteger(right.value // left.value)
        raise CSERuntimeError(f"Cannot divide {left} by {right}")

class CSEPower(CSEBinaryOp):
    def _compute(self, right: CSEValue, left: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEInteger(left.value ** right.value)
        raise CSERuntimeError(f"Cannot raise {left} to power {right}")

class CSEGreater(CSEBinaryOp):
    def _compute(self, right: CSEValue, left: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value > right.value)
        raise CSERuntimeError(f"Cannot compare {left} > {right}")

class CSEGreaterEqual(CSEBinaryOp):
    def _compute(self, right: CSEValue, left: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value >= right.value)
        raise CSERuntimeError(f"Cannot compare {left} >= {right}")

class CSELess(CSEBinaryOp):
    def _compute(self, right: CSEValue, left: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value < right.value)
        raise CSERuntimeError(f"Cannot compare {left} < {right}")

class CSELessEqual(CSEBinaryOp):
    def _compute(self, right: CSEValue, left: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value <= right.value)
        raise CSERuntimeError(f"Cannot compare {left} <= {right}")

class CSEEqual(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if type(left) != type(right):
            return CSEBoolean(False)
        
        if isinstance(left, CSEInteger):
            return CSEBoolean(left.value == right.value)
        elif isinstance(left, CSEString):
            return CSEBoolean(left.value == right.value)
        elif isinstance(left, CSEBoolean):
            return CSEBoolean(left.value == right.value)
        elif isinstance(left, CSENil) and isinstance(right, CSENil):
            return CSEBoolean(True)
        elif isinstance(left, CSEDummy) and isinstance(right, CSEDummy):
            return CSEBoolean(True)
        
        # For other types, compare by reference
        return CSEBoolean(left is right)

class CSENotEqual(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        equal_result = CSEEqual()._compute(left, right)
        return CSEBoolean(not equal_result.value)

class CSEOr(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEBoolean) and isinstance(right, CSEBoolean):
            return CSEBoolean(left.value or right.value)
        raise CSERuntimeError(f"Cannot compute logical OR of {left} and {right}")

class CSEAnd(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEBoolean) and isinstance(right, CSEBoolean):
            return CSEBoolean(left.value and right.value)
        raise CSERuntimeError(f"Cannot compute logical AND of {left} and {right}")

class CSEAug(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSETuple) and isinstance(right, CSETuple):
            return CSETuple(left.elements + right.elements)
        raise CSERuntimeError(f"Cannot augment {left} with {right}")

# --- Unary Operations ---
class CSEUnaryOp(CSEOperation):
    """Base class for unary operations"""
    def apply(self, machine: CSEMachine):
        operand = machine.stack.pop()
        result = self._compute(operand)
        machine.stack.append(result)
    
    def _compute(self, operand: CSEValue) -> CSEValue:
        raise NotImplementedError

class CSENeg(CSEUnaryOp):
    def _compute(self, operand: CSEValue) -> CSEValue:
        if isinstance(operand, CSEInteger):
            return CSEInteger(-operand.value)
        raise CSERuntimeError(f"Cannot negate {operand}")

class CSENot(CSEUnaryOp):
    def _compute(self, operand: CSEValue) -> CSEValue:
        if isinstance(operand, CSEBoolean):
            return CSEBoolean(not operand.value)
        raise CSERuntimeError(f"Cannot compute logical NOT of {operand}")

class CSERuntimeError(Exception):
    pass

# --- Main Functions ---

def evaluate_rpal(input_string: str, ast_only=False) -> CSEValue:
    """Parse, standardize, and evaluate an RPAL program"""
    if not ast_only:
        print(f"Parsing: {input_string}")
    
    ast = parse_rpal(input_string)
    
    if ast_only:
        ast.print_tree(0)
        return None
    
    print("\nAST Structure:")
    ast.print_tree(0)
    
    print("\nStandardizing...")
    st = ast_to_st(ast)
    print("\nStandardized Tree (ST) Structure:")
    st.print_tree(0)
    
    print("\nEvaluating...")
    machine = CSEMachine()
    result = machine.evaluate(st, ast_only)
    
    if not ast_only:
        print(f"Result: {result}")
    
    return result
# --- Test Programs ---
if __name__ == '__main__':
    # Check if -ast switch is provided
    ast_only = False
    file_name = None
    
    for arg in sys.argv[1:]:
        if arg == "-ast":
            ast_only = True
        else:
            file_name = arg
    
    if file_name:
        try:
            with open(file_name, 'r') as file:
                source = file.read()
                evaluate_rpal(source, ast_only)
        except Exception as e:
            print(f"Error: {e}")
    else:
        test_programs = [
            #"let greatestOfThree (x, y, z) = x > y & x > z -> x | y > x & y > z -> y | z in Print(greatestOfThree(12, 30, 18), greatestOfThree(4, 9, 6), greatestOfThree(8, 8, 8))"
            "let rec generateFibonacci(rangeStart, rangeEnd) = rangeStart > rangeEnd -> nil | generateFibonacci(rangeStart + 1, rangeEnd), print(' '), Print(fibonacci rangeStart) where rec fibonacci(n) = n eq 0 -> 0 | n eq 1 -> 1 | fibonacci(n - 1) + fibonacci(n - 2) in generateFibonacci(0, 10)"
           
            #"let f x y z t = x + y + z + t in Print (( 3 @f 4) 5 6 )",
            ##"let rec Rev S = S eq '' -> '' | (Rev(Stern S)) @Conc (Stem S ) in let Pairs (S1,S2) = P (Rev S1, Rev S2) where rec P (S1, S2) = S1 eq '' & S2 eq '' -> nil |  (fn L. P (Stern S1, Stern S2) aug ((Stem S1) @Conc (Stem S2))) nil in Print ( Pairs ('abc','def'))"
            #"let  rec OddEvenRec n = n eq 1 -> 'Odd' | n eq 0 -> 'Even' | OddEvenRec (n-2) in Print ( OddEvenRec 3)",
            #"let findMax a b c = (Isinteger a) & (Isinteger b) & (Isinteger c) -> a > b -> a > c -> a | c | c > b -> c | b | 'Error' in Print (findMax 4 6 2, findMax (-6) (-2) (-4), findMax 4 2 6, findMax 2 6 4)"
            #"let rec sumuptoSeries a b = a > b -> nil | (sumuptoSeries (a+1) b, Print ' ', Print (Sumupto a)) where rec Sumupto n = n eq 0 -> 0 | n + Sumupto (n-1) in sumuptoSeries 3 10"
            ##"let OEList T = OEListRec (T, Order T) where rec OEListRec (T, i) = i eq 0 -> nil (OEListRec (T, (i-1)) aug (OddEven (T i))) where OddEven n = (n - (n/2) * 2) eq 1 -> 'Odd' | 'Even' in Print (OEList (3, 5, 121, 10, 12))"
            #"let getGrade marks = not (Isinteger marks) -> 'Please enter an integer'| (marks > 100) or (marks < 0) -> 'Invalid Input'| marks >= 75 -> 'A'| marks >= 65 -> 'B'| marks >= 50 -> 'C'| 'F' in Print (getGrade 65)"
            #"let determineSign num = num > 0 -> 'Positive' | num < 0 -> 'Negative' | 'Zero' in Print(determineSign(8), determineSign(-2), determineSign(0))"
            #"let greatestOfThree (x, y, z) = x > y & x > z -> x | y > x & y > z -> y | z in Print(greatestOfThree(12, 30, 18), greatestOfThree(4, 9, 6), greatestOfThree(8, 8, 8))"
        ]
        

        
        for i, program in enumerate(test_programs, 1):
            print(f"\n======= Test Program {i} =======")
            try:
                evaluate_rpal(program, ast_only)
            except Exception as e:
                print(f"Error: {e}")
            print("===============================")
