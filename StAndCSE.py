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
        Recursively standardize an AST node
        """
        node_type = node.node_type
        
        if node_type == "let":
            # let D in E => (lambda X.E) D*
            d_node = node.children[0]
            e_node = node.children[1]
            
            if d_node.node_type == "=":
                # Simple binding: let X = E1 in E2 => (lambda X.E2) E1
                x_node = d_node.children[0]
                e1_node = d_node.children[1]
                
                lambda_node = STNode("lambda")
                lambda_node.add_child(self._standardize_node(x_node))
                lambda_node.add_child(self._standardize_node(e_node))
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_node)
                gamma_node.add_child(self._standardize_node(e1_node))
                return gamma_node
                
            elif d_node.node_type == "function_form":
                # Function form: let f x1 x2 ... = E1 in E2
                # => let f = lambda x1.lambda x2...E1 in E2
                # => (lambda f.E2) (lambda x1.lambda x2...E1)
                f_node = d_node.children[0]
                params = d_node.children[1:-1]
                body = d_node.children[-1]
                
                # Create nested lambdas for the function body
                current_body = self._standardize_node(body)
                for param in reversed(params):
                    lambda_inner = STNode("lambda")
                    lambda_inner.add_child(self._standardize_node(param))
                    lambda_inner.add_child(current_body)
                    current_body = lambda_inner
                
                # Create the outer lambda and gamma
                lambda_outer = STNode("lambda")
                lambda_outer.add_child(self._standardize_node(f_node))
                lambda_outer.add_child(self._standardize_node(e_node))
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_outer)
                gamma_node.add_child(current_body)
                return gamma_node
                
            elif d_node.node_type == "and":
                # Multiple bindings: let x1=e1 and x2=e2 and ... in E
                # => let x1=e1 in let x2=e2 in ... in E
                # We'll convert this to nested lambdas and applications
                
                # Extract all bindings
                bindings = []
                for child in d_node.children:
                    if child.node_type == "=":
                        var_node = child.children[0]
                        expr_node = child.children[1]
                        bindings.append((var_node, expr_node))
                
                # Start with the innermost expression
                current_expr = self._standardize_node(e_node)
                
                # Build nested let expressions from inside out
                for var_node, expr_node in reversed(bindings):
                    lambda_node = STNode("lambda")
                    lambda_node.add_child(self._standardize_node(var_node))
                    lambda_node.add_child(current_expr)
                    
                    gamma_node = STNode("gamma")
                    gamma_node.add_child(lambda_node)
                    gamma_node.add_child(self._standardize_node(expr_node))
                    
                    current_expr = gamma_node
                
                return current_expr
                
            elif d_node.node_type == "rec":
                # Recursive binding: let rec f = E1 in E2
                # => let f = Y(lambda f.E1) in E2
                # => (lambda f.E2) (Y(lambda f.E1))
                
                # Get the inner binding
                inner_binding = d_node.children[0]
                if inner_binding.node_type == "=":
                    f_node = inner_binding.children[0]
                    e1_node = inner_binding.children[1]
                elif inner_binding.node_type == "function_form":
                    # Handle function form: let rec f x1 x2 ... = E1 in E2
                    f_node = inner_binding.children[0]
                    params = inner_binding.children[1:-1]
                    body = inner_binding.children[-1]
                    
                    # Create nested lambdas for the function body
                    current_body = self._standardize_node(body)
                    for param in reversed(params):
                        lambda_inner = STNode("lambda")
                        lambda_inner.add_child(self._standardize_node(param))
                        lambda_inner.add_child(current_body)
                        current_body = lambda_inner
                    
                    e1_node = STNode("lambda")
                    for param in inner_binding.children[1:-1]:
                        e1_node.add_child(self._standardize_node(param))
                    e1_node.add_child(self._standardize_node(inner_binding.children[-1]))
                else:
                    raise StandardizationError(f"Unexpected node type in rec: {inner_binding.node_type}")
                
                # Create Y combinator application
                y_node = STNode("identifier", "Y")
                
                lambda_inner = STNode("lambda")
                lambda_inner.add_child(self._standardize_node(f_node))
                lambda_inner.add_child(self._standardize_node(e1_node))
                
                gamma_inner = STNode("gamma")
                gamma_inner.add_child(y_node)
                gamma_inner.add_child(lambda_inner)
                
                # Create the outer lambda and gamma
                lambda_outer = STNode("lambda")
                lambda_outer.add_child(self._standardize_node(f_node))
                lambda_outer.add_child(self._standardize_node(e_node))
                
                gamma_outer = STNode("gamma")
                gamma_outer.add_child(lambda_outer)
                gamma_outer.add_child(gamma_inner)
                
                return gamma_outer
            
            else:
                raise StandardizationError(f"Unexpected definition type in let: {d_node.node_type}")

        elif node_type == "where":
            # E where D => let D in E
            e_node = node.children[0]
            d_node = node.children[1]
            
            # Create a let-like structure
            if d_node.node_type == "=":
                # Simple binding: E where X = E1 => (lambda X.E) E1
                x_node = d_node.children[0]
                e1_node = d_node.children[1]
                
                lambda_node = STNode("lambda")
                lambda_node.add_child(self._standardize_node(x_node))
                lambda_node.add_child(self._standardize_node(e_node))
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_node)
                gamma_node.add_child(self._standardize_node(e1_node))
                return gamma_node
                
            elif d_node.node_type == "and":
                # Multiple bindings: E where x1=e1 and x2=e2 and ...
                # => let x1=e1 in let x2=e2 in ... in E
                
                # Extract all bindings
                bindings = []
                for child in d_node.children:
                    if child.node_type == "=":
                        var_node = child.children[0]
                        expr_node = child.children[1]
                        bindings.append((var_node, expr_node))
                
                # Start with the innermost expression
                current_expr = self._standardize_node(e_node)
                
                # Build nested let expressions from inside out
                for var_node, expr_node in reversed(bindings):
                    lambda_node = STNode("lambda")
                    lambda_node.add_child(self._standardize_node(var_node))
                    lambda_node.add_child(current_expr)
                    
                    gamma_node = STNode("gamma")
                    gamma_node.add_child(lambda_node)
                    gamma_node.add_child(self._standardize_node(expr_node))
                    
                    current_expr = gamma_node
                
                return current_expr
                
            else:
                raise StandardizationError(f"Unexpected definition type in where: {d_node.node_type}")

        elif node_type == "function_form":
            # P V1 V2 ... Vn = E => P = lambda V1.lambda V2...lambda Vn.E
            p_node = node.children[0]
            params = node.children[1:-1]
            body = node.children[-1]
            
            # Create nested lambdas for the function body
            current_body = self._standardize_node(body)
            for param in reversed(params):
                lambda_inner = STNode("lambda")
                lambda_inner.add_child(self._standardize_node(param))
                lambda_inner.add_child(current_body)
                current_body = lambda_inner
            
            equals_node = STNode("=")
            equals_node.add_child(self._standardize_node(p_node))
            equals_node.add_child(current_body)
            return equals_node

        elif node_type == "lambda":
            # fn V1 V2 ... Vn . E => lambda V1.lambda V2...lambda Vn.E
            params = node.children[:-1]
            body = node.children[-1]
            
            # Create nested lambdas
            current_body = self._standardize_node(body)
            for param in reversed(params):
                lambda_node = STNode("lambda")
                lambda_node.add_child(self._standardize_node(param))
                lambda_node.add_child(current_body)
                current_body = lambda_node
            
            return current_body

        elif node_type == "rec":
            # rec D => D where D is a binding
            d_node = node.children[0]
            
            if d_node.node_type == "=":
                # Simple binding: rec X = E => X = Y(lambda X.E)
                x_node = d_node.children[0]
                e_node = d_node.children[1]
                
                # Create Y combinator application
                y_node = STNode("identifier", "Y")
                
                lambda_node = STNode("lambda")
                lambda_node.add_child(self._standardize_node(x_node))
                lambda_node.add_child(self._standardize_node(e_node))
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(y_node)
                gamma_node.add_child(lambda_node)
                
                equals_node = STNode("=")
                equals_node.add_child(self._standardize_node(x_node))
                equals_node.add_child(gamma_node)
                return equals_node
                
            elif d_node.node_type == "function_form":
                # Function form: rec f x1 x2 ... = E
                # => f = Y(lambda f.lambda x1.lambda x2...E)
                f_node = d_node.children[0]
                params = d_node.children[1:-1]
                body = d_node.children[-1]
                
                # Create nested lambdas for the function body
                current_body = self._standardize_node(body)
                for param in reversed(params):
                    lambda_inner = STNode("lambda")
                    lambda_inner.add_child(self._standardize_node(param))
                    lambda_inner.add_child(current_body)
                    current_body = lambda_inner
                
                # Create Y combinator application
                y_node = STNode("identifier", "Y")
                
                lambda_outer = STNode("lambda")
                lambda_outer.add_child(self._standardize_node(f_node))
                lambda_outer.add_child(current_body)
                
                gamma_node = STNode("gamma")
                gamma_node.add_child(y_node)
                gamma_node.add_child(lambda_outer)
                
                equals_node = STNode("=")
                equals_node.add_child(self._standardize_node(f_node))
                equals_node.add_child(gamma_node)
                return equals_node
                
            else:
                raise StandardizationError(f"Unexpected node type in rec: {d_node.node_type}")

        elif node_type == "and":
            # D1 and D2 ... => delta(X1, X2, ..., E1, E2, ...)
            # For simplicity, we'll just standardize each definition
            and_node = STNode("and")
            for child in node.children:
                and_node.add_child(self._standardize_node(child))
            return and_node

        elif node_type == "within":
            # D1 within D2 => D2 where D1
            d1_node = node.children[0]
            d2_node = node.children[1]
            
            if d1_node.node_type == "=" and d2_node.node_type == "=":
                # Simple case: X1 = E1 within X2 = E2 => X2 = E2 where X1 = E1
                x1_node = d1_node.children[0]
                e1_node = d1_node.children[1]
                x2_node = d2_node.children[0]
                e2_node = d2_node.children[1]
                
                # Create a lambda for the binding
                lambda_node = STNode("lambda")
                lambda_node.add_child(self._standardize_node(x1_node))
                
                # The body is the second definition
                equals_node = STNode("=")
                equals_node.add_child(self._standardize_node(x2_node))
                equals_node.add_child(self._standardize_node(e2_node))
                lambda_node.add_child(equals_node)
                
                # Apply the lambda to the first expression
                gamma_node = STNode("gamma")
                gamma_node.add_child(lambda_node)
                gamma_node.add_child(self._standardize_node(e1_node))
                return gamma_node
                
            else:
                # More complex cases would need special handling
                raise StandardizationError(f"Complex within not fully implemented: {d1_node.node_type} within {d2_node.node_type}")

        elif node_type == "tau":
            # (E1, E2, ..., En) => tau E1 E2 ... En
            tau_node = STNode("tau")
            for child in node.children:
                tau_node.add_child(self._standardize_node(child))
            return tau_node

        elif node_type == "->":
            # B -> E1 | E2 => -> B E1 E2
            cond_node = STNode("->")
            cond_node.add_child(self._standardize_node(node.children[0]))  # Condition
            cond_node.add_child(self._standardize_node(node.children[1]))  # Then branch
            cond_node.add_child(self._standardize_node(node.children[2]))  # Else branch
            return cond_node

        elif node_type == "gamma":
            # E1 E2 => gamma E1 E2
            gamma_node = STNode("gamma")
            gamma_node.add_child(self._standardize_node(node.children[0]))
            gamma_node.add_child(self._standardize_node(node.children[1]))
            return gamma_node

        # --- Leaf Nodes & Simple Structures ---
        elif node_type in ["identifier", "integer", "string", "true", "false", "nil", "dummy"]:
            return STNode(node_type, node.value)
            
        elif node_type == "=":
            # X = E => = X E
            equals_node = STNode("=")
            equals_node.add_child(self._standardize_node(node.children[0]))
            equals_node.add_child(self._standardize_node(node.children[1]))
            return equals_node

        # --- Operators ---
        elif node_type in ["aug", "or", "&", "not", "gr", "ge", "ls", "le", "eq", "ne", "+", "-", "*", "/", "**", "neg", "@"]:
            op_node = STNode(node_type)
            for child in node.children:
                op_node.add_child(self._standardize_node(child))
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
    def __str__(self): return f"({', '.join(map(str, self.elements))})"
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
            elements = []
            for element in arg.elements:
                elements.append(str(element))
            print(" ".join(elements))
        else:
            # Print a single value
            print(arg)
        
        # Return dummy
        machine.stack.append(CSEDummy())

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
        
        # Initialize built-in functions
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in functions"""
        self.env.bind("Y", CSEYCombinator())
        self.env.bind("Print", CSEPrint())
        # Add other built-ins as needed
    
    def evaluate(self, st_node: STNode, ast_only=False) -> CSEValue:
        """Evaluate a standardized tree"""
        if ast_only:
            return None
            
        self.control = [st_node]
        self.stack = []
        self.env_stack = []
        
        while self.control:
            # Debug output
            # print(f"Control: {[str(c) for c in self.control]}")
            # print(f"Stack: {[str(s) for s in self.stack]}")
            # print(f"Env: {self.env}")
            # print("---")
            
            item = self.control.pop()
            
            if isinstance(item, STNode):
                self._process_node(item)
            elif isinstance(item, CSEOperation):
                item.apply(self)
            else:
                raise CSERuntimeError(f"Invalid control item: {item}")
        
        if not self.stack:
            return CSENil()
        return self.stack[0]
    
    def _process_node(self, node: STNode):
        """Process a node from the control stack"""
        node_type = node.node_type
        
        if node_type == "lambda":
            # Create a closure with the current environment
            var_name = node.children[0].value
            body = node.children[1]
            closure = CSEClosure(var_name, body, self.env)
            self.stack.append(closure)
            
        elif node_type == "gamma":
            # Function application
            self.control.append(CSEApply())
            self.control.append(node.children[0])  # Function
            self.control.append(node.children[1])  # Argument
            
        elif node_type == "identifier":
            # Look up the identifier in the environment
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
            # Create a tuple
            n = len(node.children)
            self.control.append(CSETupleConstructor(n))
            for child in reversed(node.children):
                self.control.append(child)
                
        elif node_type == "->":
            # Conditional
            condition = node.children[0]
            then_branch = node.children[1]
            else_branch = node.children[2]
            self.control.append(CSEConditional(then_branch, else_branch))
            self.control.append(condition)
            
        elif node_type in ["+", "-", "*", "/", "**"]:
            # Binary arithmetic operations
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
            # Comparison operations
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
            # Boolean operations
            op_map = {
                "or": CSEOr(),
                "&": CSEAnd()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            self.control.append(node.children[1])
            
        elif node_type in ["neg", "not"]:
            # Unary operations
            op_map = {
                "neg": CSENeg(),
                "not": CSENot()
            }
            self.control.append(op_map[node_type])
            self.control.append(node.children[0])
            
        elif node_type == "aug":
            # Tuple augmentation
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
            
            # Bind the parameter to the argument
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
            return CSEInteger(left.value - right.value)
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
            return CSEInteger(left.value // right.value)
        raise CSERuntimeError(f"Cannot divide {left} by {right}")

class CSEPower(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEInteger(left.value ** right.value)
        raise CSERuntimeError(f"Cannot raise {left} to power {right}")

class CSEGreater(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value > right.value)
        raise CSERuntimeError(f"Cannot compare {left} > {right}")

class CSEGreaterEqual(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value >= right.value)
        raise CSERuntimeError(f"Cannot compare {left} >= {right}")

class CSELess(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
        if isinstance(left, CSEInteger) and isinstance(right, CSEInteger):
            return CSEBoolean(left.value < right.value)
        raise CSERuntimeError(f"Cannot compare {left} < {right}")

class CSELessEqual(CSEBinaryOp):
    def _compute(self, left: CSEValue, right: CSEValue) -> CSEValue:
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
            # Simple arithmetic
            "Print(1 + 2 * 3)",
            
            # Simple let expression
            "let X = 3 in Print(X)",
            
            # Function application
            "let f x = x + 1 in Print(f(5))",
            
            # Conditional
            "let Abs N = N ls 0 -> -N | N in Print(Abs(-3))",
            
            # Recursive function
            "let rec fact n = n eq 0 -> 1 | n * fact(n-1) in Print(fact(5))",
            
            # Multi-parameter function
            "let add x y = x + y in Print(add(3, 4))",
            
            # Tuple
            "let T = 1, 2, 3 in Print(T)",
            
            # Where clause
            "Print(X + Y) where X = 3 and Y = 4",
            
            # Lambda expression
            "let double = fn x. x * 2 in Print(double(5))",
            
            # Boolean operations
            "Print(true & false or true)"
        ]
        
        for i, program in enumerate(test_programs, 1):
            print(f"\n======= Test Program {i} =======")
            try:
                evaluate_rpal(program, ast_only)
            except Exception as e:
                print(f"Error: {e}")
            print("===============================")
