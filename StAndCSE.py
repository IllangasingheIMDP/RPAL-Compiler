import sys
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from parser import ASTNode, parse_rpal
from lexical import rpal_lexer

# --- Standardized Tree Node ---
class STNode:
    """
    Node in the Standardized Tree (ST) for RPAL
    """
    def __init__(self, node_type: str, value: Any = None):
        self.node_type = node_type
        self.value = value
        self.children = []
    
    def add_child(self, child):
        """Add a child node to this node"""
        self.children.append(child)
    
    def __str__(self):
        """String representation of the node"""
        if self.value is not None:
            # Special handling for identifiers and strings for clarity
            if self.node_type == "identifier":
                return f"<ID:{self.value}>"
            elif self.node_type == "integer":
                return f"<INT:{self.value}>"
            elif self.node_type == "string":
                return f"<STR:{self.value}>"
            return f"<{self.node_type}:{self.value}>"
        return f"<{self.node_type}>"
    
    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent=0):
        """Print the tree with indentation"""
        print("." * indent + str(self))
        for child in self.children:
            if isinstance(child, STNode):
                child.print_tree(indent + 1)
            else:
                # Handle non-node children if any (shouldn't happen in ST)
                print("." * (indent + 1) + str(child))

# --- AST to ST Conversion ---
class StandardizationEngine:
    """
    Engine to convert AST to Standardized Tree (ST)
    """
    def __init__(self):
        self.var_counter = 0
    
    def standardize(self, ast_node: ASTNode) -> STNode:
        """
        Convert an AST to a Standardized Tree (ST)
        """
        return self._standardize_node(ast_node)
    
    def _standardize_node(self, node: ASTNode) -> STNode:
        """
        Recursively standardize an AST node using a bottom-up approach
        """
        node_type = node.node_type
        
        # First standardize all children (bottom-up approach)
        standardized_children = []
        for child in node.children:
            standardized_children.append(self._standardize_node(child))
        
        if node_type == "let":
            # let D in E => (lambda X.E) D*
            d_node = node.children[0]
            e_node = node.children[1]
            
            # Use the standardized children
            std_d_node = standardized_children[0]
            std_e_node = standardized_children[1]
            
            if d_node.node_type == "=":
                # Simple binding: let X = E1 in E2 => (lambda X.E2) E1
                x_node = std_d_node.children[0]  # X
                e1_node = std_d_node.children[1]  # E1
                
                lambda_node = STNode("lambda")
                lambda_node.add_child(x_node)
                lambda_node.add_child(std_e_node)  # E2
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_node)
                gamma_node.add_child(e1_node)
                return gamma_node
                
            elif d_node.node_type == "function_form":
                # Function form: let f x1 x2 ... = E1 in E2
                # => let f = lambda x1.lambda x2...E1 in E2
                # => (lambda f.E2) (lambda x1.lambda x2...E1)
                f_node = std_d_node.children[0]  # f
                
                # Create nested lambdas for the function body
                # Extract parameters and body from standardized function form
                params = []
                for i in range(1, len(std_d_node.children) - 1):
                    params.append(std_d_node.children[i])
                body = std_d_node.children[-1]
                
                # Create nested lambdas
                current_body = body
                for param in reversed(params):
                    lambda_inner = STNode("lambda")
                    lambda_inner.add_child(param)
                    lambda_inner.add_child(current_body)
                    current_body = lambda_inner
                
                # Create the outer lambda and gamma
                lambda_outer = STNode("lambda")
                lambda_outer.add_child(f_node)
                lambda_outer.add_child(std_e_node)
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_outer)
                gamma_node.add_child(current_body)
                return gamma_node
                
            elif d_node.node_type == "and":
                # Multiple bindings: let x1=e1 and x2=e2 and ... in E
                # => let x1=e1 in let x2=e2 in ... in E
                
                # Extract all bindings from the standardized 'and' node
                bindings = []
                for child in std_d_node.children:
                    if child.node_type == "=":
                        var_node = child.children[0]
                        expr_node = child.children[1]
                        bindings.append((var_node, expr_node))
                
                # Start with the innermost expression
                current_expr = std_e_node
                
                # Build nested let expressions from inside out
                for var_node, expr_node in reversed(bindings):
                    lambda_node = STNode("lambda")
                    lambda_node.add_child(var_node)
                    lambda_node.add_child(current_expr)
                    
                    gamma_node = STNode("gamma")
                    gamma_node.add_child(lambda_node)
                    gamma_node.add_child(expr_node)
                    
                    current_expr = gamma_node
                
                return current_expr
                
            elif d_node.node_type == "rec":
                # Recursive binding: let rec f = E1 in E2
                # => let f = Y(lambda f.E1) in E2
                # => (lambda f.E2) (Y(lambda f.E1))
                
                # The standardized rec node already has the Y combinator structure
                
                # Get the function name and body from the standardized rec node
                if std_d_node.node_type == "=" and len(std_d_node.children) >= 2:
                    f_node = std_d_node.children[0]  # From '=' node
                    y_expr = std_d_node.children[1]  # Y(...) from '=' node
                    
                    # Create the outer lambda and gamma
                    lambda_outer = STNode("lambda")
                    lambda_outer.add_child(f_node)
                    lambda_outer.add_child(std_e_node)
                    
                    gamma_outer = STNode("gamma")
                    gamma_outer.add_child(lambda_outer)
                    gamma_outer.add_child(y_expr)
                    
                    return gamma_outer
                else:
                    # Handle the case where rec node has a different structure
                    # This is a fallback for unexpected structures
                    y_node = STNode("identifier", "Y")
                    
                    # Extract function name and create a lambda
                    if len(std_d_node.children) > 0:
                        f_node = std_d_node.children[0]
                        
                        # Create a lambda for the recursive function
                        lambda_inner = STNode("lambda")
                        lambda_inner.add_child(f_node)
                        
                        # If there's a body, use it, otherwise create a simple identity function
                        if len(std_d_node.children) > 1:
                            body_node = std_d_node.children[1]
                            lambda_inner.add_child(body_node)
                        else:
                            # Identity function as fallback
                            lambda_inner.add_child(f_node)
                        
                        gamma_inner = STNode("gamma")
                        gamma_inner.add_child(y_node)
                        gamma_inner.add_child(lambda_inner)
                        
                        # Create the outer lambda and gamma
                        lambda_outer = STNode("lambda")
                        lambda_outer.add_child(f_node)
                        lambda_outer.add_child(std_e_node)
                        
                        gamma_outer = STNode("gamma")
                        gamma_outer.add_child(lambda_outer)
                        gamma_outer.add_child(gamma_inner)
                        
                        return gamma_outer
                    else:
                        # If we can't extract anything useful, just return a simple binding
                        lambda_node = STNode("lambda")
                        dummy_id = STNode("identifier", "dummy")
                        lambda_node.add_child(dummy_id)
                        lambda_node.add_child(std_e_node)
                        
                        gamma_node = STNode("gamma")
                        gamma_node.add_child(lambda_node)
                        gamma_node.add_child(STNode("nil"))
                        return gamma_node
            
            else:
                raise StandardizationError(f"Unexpected definition type in let: {d_node.node_type}")

        elif node_type == "where":
            # E where D => let D in E
            e_node = node.children[0]
            d_node = node.children[1]
            
            # Use the standardized children
            std_e_node = standardized_children[0]
            std_d_node = standardized_children[1]
            
            if d_node.node_type == "=":
                # Check if left side is a comma node (tuple unpacking)
                if d_node.children[0].node_type == ",":
                    # For tuple unpacking: E where x,y = (1,2)
                    # We need to create a special node to handle this
                    
                    # Extract variable names from the comma node
                    var_nodes = []
                    for child in d_node.children[0].children:
                        if child.node_type == "identifier":
                            var_nodes.append(self._standardize_node(child))
                    
                    # Get the tuple expression (right side)
                    tuple_expr = std_d_node.children[1]
                    
                    # Create a special tuple unpacking node
                    unpack_node = STNode("tuple_unpack")
                    for var_node in var_nodes:
                        unpack_node.add_child(var_node)
                    
                    # Create a lambda for each variable
                    current_expr = std_e_node
                    for var_node in reversed(var_nodes):
                        lambda_node = STNode("lambda")
                        lambda_node.add_child(var_node)
                        lambda_node.add_child(current_expr)
                        current_expr = lambda_node
                    
                    # Create a special gamma node for tuple unpacking
                    gamma_node = STNode("tuple_apply")
                    gamma_node.add_child(current_expr)
                    gamma_node.add_child(tuple_expr)
                    gamma_node.add_child(unpack_node)  # Add the unpacking info
                    
                    return gamma_node
                else:
                    # Simple binding: E where X = E1 => (lambda X.E) E1
                    x_node = std_d_node.children[0]
                    e1_node = std_d_node.children[1]
                    
                    lambda_node = STNode("lambda")
                    lambda_node.add_child(x_node)
                    lambda_node.add_child(std_e_node)
                    
                    gamma_node = STNode("gamma")
                    gamma_node.add_child(lambda_node)
                    gamma_node.add_child(e1_node)
                    return gamma_node
                
            elif d_node.node_type == "and":
                # Multiple bindings: E where x1=e1 and x2=e2 and ...
                # => let x1=e1 in let x2=e2 in ... in E
                
                # Extract all bindings from the standardized 'and' node
                bindings = []
                for child in std_d_node.children:
                    if child.node_type == "=":
                        var_node = child.children[0]
                        expr_node = child.children[1]
                        bindings.append((var_node, expr_node))
                
                # Start with the innermost expression
                current_expr = std_e_node
                
                # Build nested let expressions from inside out
                for var_node, expr_node in reversed(bindings):
                    lambda_node = STNode("lambda")
                    lambda_node.add_child(var_node)
                    lambda_node.add_child(current_expr)
                    
                    gamma_node = STNode("gamma")
                    gamma_node.add_child(lambda_node)
                    gamma_node.add_child(expr_node)
                    
                    current_expr = gamma_node
                
                return current_expr
                
            elif d_node.node_type == "rec":
                # Handle recursive definition in where clause
                # E where rec D => let rec D in E
                
                # Get the function name and body from the standardized rec node
                if std_d_node.node_type == "=" and len(std_d_node.children) >= 2:
                    f_node = std_d_node.children[0]  # Function name
                    y_expr = std_d_node.children[1]  # Y(...) expression
                    
                    # Create the outer lambda and gamma
                    lambda_outer = STNode("lambda")
                    lambda_outer.add_child(f_node)
                    lambda_outer.add_child(std_e_node)
                    
                    gamma_outer = STNode("gamma")
                    gamma_outer.add_child(lambda_outer)
                    gamma_outer.add_child(y_expr)
                    
                    return gamma_outer
                else:
                    # Handle the case where rec node has a different structure
                    # This is a fallback for unexpected structures
                    y_node = STNode("identifier", "Y")
                    
                    # Extract function name and create a lambda
                    if len(std_d_node.children) > 0:
                        f_node = std_d_node.children[0]
                        
                        # Create a lambda for the recursive function
                        lambda_inner = STNode("lambda")
                        lambda_inner.add_child(f_node)
                        
                        # If there's a body, use it, otherwise create a simple identity function
                        if len(std_d_node.children) > 1:
                            body_node = std_d_node.children[1]
                            lambda_inner.add_child(body_node)
                        else:
                            # Identity function as fallback
                            lambda_inner.add_child(f_node)
                        
                        gamma_inner = STNode("gamma")
                        gamma_inner.add_child(y_node)
                        gamma_inner.add_child(lambda_inner)
                        
                        # Create the outer lambda and gamma
                        lambda_outer = STNode("lambda")
                        lambda_outer.add_child(f_node)
                        lambda_outer.add_child(std_e_node)
                        
                        gamma_outer = STNode("gamma")
                        gamma_outer.add_child(lambda_outer)
                        gamma_outer.add_child(gamma_inner)
                        
                        return gamma_outer
                    else:
                        # If we can't extract anything useful, just return a simple binding
                        lambda_node = STNode("lambda")
                        dummy_id = STNode("identifier", "dummy")
                        lambda_node.add_child(dummy_id)
                        lambda_node.add_child(std_e_node)
                        
                        gamma_node = STNode("gamma")
                        gamma_node.add_child(lambda_node)
                        gamma_node.add_child(STNode("nil"))
                        return gamma_node
            
            else:
                raise StandardizationError(f"Unexpected definition type in where: {d_node.node_type}")

        elif node_type == "function_form":
            # P V1 V2 ... Vn = E => P = lambda V1.lambda V2...lambda Vn.E
            p_node = standardized_children[0]  # Function name
            params = []
            
            # Extract parameters, handling comma nodes
            for i in range(1, len(standardized_children) - 1):
                child = standardized_children[i]
                if child.node_type == ",":
                    # If it's a comma node, extract its children as parameters
                    for comma_child in child.children:
                        params.append(comma_child)
                else:
                    params.append(child)
            
            body = standardized_children[-1]  # Function body
            
            # Create nested lambdas for the function body
            current_body = body
            for param in reversed(params):
                lambda_inner = STNode("lambda")
                lambda_inner.add_child(param)
                lambda_inner.add_child(current_body)
                current_body = lambda_inner
            
            # Create an equals node for the function definition
            equals_node = STNode("=")
            equals_node.add_child(p_node)
            equals_node.add_child(current_body)
            
            return equals_node

        elif node_type == "lambda":
            # lambda X.E => lambda X E
            lambda_node = STNode("lambda")
            for std_child in standardized_children:
                lambda_node.add_child(std_child)
            return lambda_node

        elif node_type == "within":
            # E1 within E2 => let X = E2 in E1[X/E2]
            # where X is a fresh variable
            
            e1_node = node.children[0]  # Expression using the definition
            e2_node = node.children[1]  # Definition
            
            # Use the standardized children
            std_e1_node = standardized_children[0]
            std_e2_node = standardized_children[1]
            
            if e2_node.node_type == "=":
                # Simple binding: E1 within X = E2 => let X = E2 in E1
                x_node = std_e2_node.children[0]  # X
                e2_expr_node = std_e2_node.children[1]  # E2 expression
                
                lambda_node = STNode("lambda")
                lambda_node.add_child(x_node)
                lambda_node.add_child(std_e1_node)
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_node)
                gamma_node.add_child(e2_expr_node)
                return gamma_node
                
            else:
                # More complex cases would need special handling
                raise StandardizationError(f"Complex within not fully implemented: {node.children[0].node_type} within {node.children[1].node_type}")

        elif node_type == "tau":
            # (E1, E2, ..., En) => tau E1 E2 ... En
            tau_node = STNode("tau")
            for std_child in standardized_children:
                tau_node.add_child(std_child)
            return tau_node

        elif node_type == "->":
            # B -> E1 | E2 => -> B E1 E2
            cond_node = STNode("->")
            for std_child in standardized_children:
                cond_node.add_child(std_child)
            return cond_node

        elif node_type == "gamma":
            # E1 E2 => gamma E1 E2
            gamma_node = STNode("gamma")
            for std_child in standardized_children:
                gamma_node.add_child(std_child)
            return gamma_node

        # --- Leaf Nodes & Simple Structures ---
        elif node_type in ["identifier", "integer", "string", "true", "false", "nil", "dummy"]:
            return STNode(node_type, node.value)
            
        elif node_type == "=":
            # X = E => = X E
            equals_node = STNode("=")
            for std_child in standardized_children:
                equals_node.add_child(std_child)
            return equals_node

        # --- Handle comma node for parameter lists ---
        elif node_type == ",":
            # For parameter lists in function definitions
            comma_node = STNode(",")
            for std_child in standardized_children:
                comma_node.add_child(std_child)
            return comma_node

        # --- Operators ---
        elif node_type in ["aug", "or", "&", "not", "gr", "ge", "ls", "le", "eq", "ne", "+", "-", "*", "/", "**", "neg", "@"]:
            op_node = STNode(node_type)
            for std_child in standardized_children:
                op_node.add_child(std_child)
            return op_node
            
        else:
            raise StandardizationError(f"Unknown AST node type during standardization: {node_type}")

    def _fresh_var(self, prefix="x"):
        """Generate a fresh variable name"""
        self.var_counter += 1
        return f"{prefix}{self.var_counter}"

class StandardizationError(Exception):
    pass

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
        self.debug = True  # Enable debug output
        
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
            if self.debug:
                print(f"\n--- Step {step} ---")
                print("Control:", [str(c) for c in self.control])
                print("Stack:", [str(s) for s in self.stack])
                print("Env:", self.env)
            item = self.control.pop()
            if self.debug:
                print("Popped from control:", item)
            if isinstance(item, STNode):
                if self.debug:
                    print("Processing node:", item)
                self._process_node(item)
            elif isinstance(item, CSEOperation):
                if self.debug:
                    print("Processing operation:", item)
                item.apply(self)
            else:
                raise CSERuntimeError(f"Invalid control item: {item}")
            if self.debug:
                print("Stack after step:", [str(s) for s in self.stack])
                print("Env after step:", self.env)
            step += 1
        
        if not self.stack:
            return CSENil()
        return self.stack[0]
    
    def _process_node(self, node: STNode):
        """Process a node from the control stack"""
        node_type = node.node_type
        
        if self.debug:
            print(f"Processing node type: {node_type}")
        
        if node_type == "lambda":
            var_name = node.children[0].value
            body = node.children[1]
            closure = CSEClosure(var_name, body, self.env)
            if self.debug:
                print(f"Created closure for var '{var_name}' with env {self.env}")
            self.stack.append(closure)
            
        elif node_type == "gamma":
            if self.debug:
                print("Preparing function application (gamma)")
            self.control.append(CSEApply())
            self.control.append(node.children[0])  # Function
            self.control.append(node.children[1])  # Argument
            
        elif node_type == "tuple_apply":
            if self.debug:
                print("Preparing tuple unpacking application (tuple_apply)")
            self.control.append(CSETupleApply(node.children[2]))  # Unpacking info
            self.control.append(node.children[0])  # Function (lambda chain)
            self.control.append(node.children[1])  # Tuple expression
            
        elif node_type == "identifier":
            if self.debug:
                print(f"Looking up identifier: {node.value}")
            value = self.env.lookup(node.value)
            if self.debug:
                print(f"Looked up identifier '{node.value}': {value}")
            self.stack.append(value)
            
        elif node_type == "integer":
            if self.debug:
                print(f"Pushing integer: {node.value}")
            self.stack.append(CSEInteger(node.value))
            
        elif node_type == "string":
            if self.debug:
                print(f"Pushing string: {node.value}")
            self.stack.append(CSEString(node.value))
            
        elif node_type == "true":
            if self.debug:
                print("Pushing boolean: true")
            self.stack.append(CSEBoolean(True))
            
        elif node_type == "false":
            if self.debug:
                print("Pushing boolean: false")
            self.stack.append(CSEBoolean(False))
            
        elif node_type == "nil":
            if self.debug:
                print("Pushing nil")
            self.stack.append(CSENil())
            
        elif node_type == "dummy":
            if self.debug:
                print("Pushing dummy")
            self.stack.append(CSEDummy())
            
        elif node_type == "tau":
            n = len(node.children)
            if self.debug:
                print(f"Preparing tuple of size {n}")
            self.control.append(CSETupleConstructor(n))
            for child in reversed(node.children):
                self.control.append(child)
                
        elif node_type == "->":
            condition = node.children[0]
            then_branch = node.children[1]
            else_branch = node.children[2]
            if self.debug:
                print("Preparing conditional (->)")
            self.control.append(CSEConditional(then_branch, else_branch))
            self.control.append(condition)
            
        elif node_type in ["+", "-", "*", "/", "**"]:
            if self.debug:
                print(f"Preparing binary arithmetic operation: {node_type}")
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
            if self.debug:
                print(f"Preparing comparison operation: {node_type}")
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
            if self.debug:
                print(f"Preparing boolean operation: {node_type}")
            op_map = {
                "or": CSEOr(),
                "&": CSEAnd()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        elif node_type in ["neg", "not"]:
            if self.debug:
                print(f"Preparing unary operation: {node_type}")
            op_map = {
                "neg": CSENeg(),
                "not": CSENot()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            
        elif node_type == "aug":
            if self.debug:
                print("Preparing tuple augmentation (aug)")
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

class CSEApply(CSEOperation):
    """Apply operation for function application"""
    def apply(self, machine: CSEMachine):
        # Pop function and argument
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
def ast_to_st(ast_node: ASTNode) -> STNode:
    """Convert an AST to a Standardized Tree"""
    engine = StandardizationEngine()
    return engine.standardize(ast_node)

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
            "let getGrade marks = not (Isinteger marks) -> 'Please enter an integer'| (marks > 100) or (marks < 0) -> 'Invalid Input'| marks >= 75 -> 'A'| marks >= 65 -> 'B'| marks >= 50 -> 'C'| 'F' in Print (getGrade 65)"
          
        ]
        #"let getGrade marks = not (Isinteger marks) -> 'Please enter an integer'| (marks > 100) or (marks < 0) -> 'Invalid Input'| marks >= 75 -> 'A'| marks >= 65 -> 'B'| marks >= 50 -> 'C'| 'Fin Print (getGrade 65)"

        
        for i, program in enumerate(test_programs, 1):
            print(f"\n======= Test Program {i} =======")
            try:
                evaluate_rpal(program, ast_only)
            except Exception as e:
                print(f"Error: {e}")
            print("===============================")
