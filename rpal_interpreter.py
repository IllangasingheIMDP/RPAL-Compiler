"""
RPAL Interpreter Engine - Core Implementation

This module implements a Control Structure Execution (CSE) engine for interpreting RPAL programs.
It provides the core functionality for executing standardized trees generated from RPAL source code.

The implementation follows a stack-based execution model with environment management for variable scoping.
"""

from collections import defaultdict
import functools

#==============================================================================
# SECTION 1: OPERATION HANDLERS
#==============================================================================

def execute_unary_operation(interpreter, operand, operation_name):
    """
    Executes a unary operation on a single operand.
    
    Args:
        interpreter: The interpreter instance
        operand: The value to operate on
        operation_name: Name of the unary operation to perform
        
    Returns:
        Result of the unary operation
    """
    # Extract the operand value
    operand_value = operand.value

    # Dictionary of all supported unary operations
    unary_operation_map = {
        "Print"       : lambda interp, op: handle_print_operation(interp, op),
        "Isstring"    : lambda interp, op: op.type == "STR",
        "Isinteger"   : lambda interp, op: op.type == "INT",
        "Istruthvalue": lambda interp, op: op.type == "bool",
        "Isfunction"  : lambda interp, op: op.type == "lambda",
        "Null"        : lambda interp, op: op.type == "nil",
        "Istuple"     : lambda interp, op: isinstance(op.value, list) or op.type == "nil",
        "Order"       : lambda interp, op: handle_order_operation(interp, op),
        "Stern"       : lambda interp, op: handle_stern_operation(interp, op.value),
        "Stem"        : lambda interp, op: handle_stem_operation(interp, op.value),
        "ItoS"        : lambda interp, op: str(op.value) if isinstance(op.value, int) and not isinstance(op.value, bool) else interp._error_manager.report_error("Interpreter: Invalid type for ItoS operation"),
        "neg"         : lambda interp, op: -op.value if isinstance(op.value, int) else interp._error_manager.report_error("Interpreter: Invalid type for negation"),
        "not"         : lambda interp, op: not op.value if isinstance(op.value, bool) else interp._error_manager.report_error("Interpreter: Invalid type for logical NOT"),
    }
    
    # Look up and execute the operation
    operation_function = unary_operation_map.get(operation_name)
    if operation_function:
        return operation_function(interpreter, operand)
    else:
        raise ValueError(f"Unknown unary operation: {operation_name}")

def handle_print_operation(interpreter, operand):
    """
    Handles the Print operation by converting any value to its string representation.
    
    Args:
        interpreter: The interpreter instance
        operand: The value to be printed
        
    Returns:
        'dummy' as a placeholder (actual output is queued)
    """
    element = operand.value
    
    # Helper function for converting values to strings
    def value_to_string(element):
        if isinstance(element, list):
            output = ""
            return convert_tuple_to_string(element, output)
        elif element == "lambda":
            param_list = "".join(x for x in operand.bounded_vars)
            ctrl_idx = str(operand.control_idx)
            return f"[lambda closure: {param_list}: {ctrl_idx}]"
        elif isinstance(element, bool):
            return "true" if element else "false"
        elif isinstance(element, str):
            return element
        elif isinstance(element, int):
            return str(element)
        elif element is None:
            return "nil"
        else:
            raise TypeError("Unknown element type in print operation.")
        
    def convert_tuple_to_string(element, output):
        if isinstance(element, list):
            output += "("
            for item in element:
                output = convert_tuple_to_string(item, output)
            output = output[:-2] + ")"
        else:
            if isinstance(element.value, list):
                output += "("
                for item in element.value:
                    output = convert_tuple_to_string(item, output)
                output = output[:-2] + "), "
            else:
                output += value_to_string(element.value) + ", "
        return output
    
    # Add the formatted string to the output queue with escape sequence handling
    interpreter._output_queue.append(value_to_string(element).replace("\\n", "\n").replace("\\t", "\t"))
    
    return "dummy"

def handle_order_operation(interpreter, operand):
    """
    Implements the Order operation which returns the length of a tuple.
    
    Args:
        interpreter: The interpreter instance
        operand: The tuple to find the length of
        
    Returns:
        Integer length of the tuple or 0 for nil
    """
    if isinstance(operand.value, list):
        return len(operand.value)
    elif operand.type == "nil":
        return 0
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid type for Order operation")

def handle_stern_operation(interpreter, operand):
    """
    Implements the Stern operation which returns all but the first character of a string.
    
    Args:
        interpreter: The interpreter instance
        operand: The string to get the stern of
        
    Returns:
        String without its first character
    """
    if isinstance(operand, str) and len(operand) >= 1:
        return operand[1:]
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid type for Stern operation")

def handle_stem_operation(interpreter, operand):
    """
    Implements the Stem operation which returns the first character of a string.
    
    Args:
        interpreter: The interpreter instance
        operand: The string to get the stem of
        
    Returns:
        First character of the string
    """
    if isinstance(operand, str) and len(operand) >= 1:
        return operand[0]
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid type for Stem operation")

def execute_binary_operation(interpreter, left_operand, right_operand, operation_name):
    """
    Executes a binary operation on two operands.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First operand
        right_operand: Second operand
        operation_name: Name of the binary operation
        
    Returns:
        Result of the binary operation
    """
    # Dictionary of all supported binary operations
    binary_operation_map = {
        "aug" : lambda interp, left, right: handle_augment_operation(interp, left, right),
        "or"  : lambda interp, left, right: handle_logical_or(interp, left, right),
        "&"   : lambda interp, left, right: handle_logical_and(interp, left, right),
        "+"   : lambda interp, left, right: handle_arithmetic(interp, left, right, lambda a, b: a + b),
        "-"   : lambda interp, left, right: handle_arithmetic(interp, left, right, lambda a, b: a - b),
        "*"   : lambda interp, left, right: handle_arithmetic(interp, left, right, lambda a, b: a * b),
        "/"   : lambda interp, left, right: handle_arithmetic(interp, left, right, lambda a, b: a // b),
        "**"  : lambda interp, left, right: handle_arithmetic(interp, left, right, lambda a, b: a ** b),
        "gr"  : lambda interp, left, right: handle_comparison(interp, left, right, lambda a, b: a > b),
        "ge"  : lambda interp, left, right: handle_comparison(interp, left, right, lambda a, b: a >= b),
        "ls"  : lambda interp, left, right: handle_comparison(interp, left, right, lambda a, b: a < b),
        "le"  : lambda interp, left, right: handle_comparison(interp, left, right, lambda a, b: a <= b),
        "eq"  : lambda interp, left, right: handle_equality(interp, left, right),
        "ne"  : lambda interp, left, right: handle_inequality(interp, left, right),
        "Conc": lambda interp, left, right: handle_concatenation(interp, left, right)
    }

    operation_function = binary_operation_map.get(operation_name)
    if operation_function:
        return operation_function(interpreter, left_operand, right_operand)
    else:
        raise ValueError(f"Unknown binary operation: {operation_name}")

def handle_augment_operation(interpreter, left_operand, right_operand):
    """
    Implements tuple augmentation (aug) which combines elements into tuples.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First operand (left side of aug)
        right_operand: Second operand (right side of aug)
        
    Returns:
        A new tuple containing the combined elements
    """
    if left_operand.type == "nil":
        return ControlElement("tuple", [right_operand])
    elif right_operand.type == "nil":
        if left_operand.type == "tuple":
            left_operand.value.append(right_operand)
            return left_operand
        return left_operand
    elif left_operand.type in ["tuple", "ID", "INT", "STR", "bool"] and right_operand.type in ["tuple", "ID", "INT", "STR", "bool"]:
        if isinstance(left_operand.value, list):
            elements = left_operand.value.copy()
            elements.append(right_operand)
            return ControlElement("tuple", elements)
        return ControlElement("tuple", [left_operand, right_operand])
    else:
        interpreter._error_manager.report_error("Interpreter: Cannot augment a non-tuple value")

def handle_logical_or(interpreter, left_operand, right_operand):
    """
    Implements logical OR operation between two boolean values.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First boolean operand
        right_operand: Second boolean operand
        
    Returns:
        Boolean result of OR operation
    """
    if isinstance(left_operand, bool) and isinstance(right_operand, bool):
        return left_operand or right_operand
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid types for logical OR")

def handle_logical_and(interpreter, left_operand, right_operand):
    """
    Implements logical AND operation between two boolean values.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First boolean operand
        right_operand: Second boolean operand
        
    Returns:
        Boolean result of AND operation
    """
    if isinstance(left_operand, bool) and isinstance(right_operand, bool):
        return left_operand and right_operand
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid types for logical AND")

def handle_equality(interpreter, left_operand, right_operand):
    """
    Implements equality comparison between two values of the same type.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First operand
        right_operand: Second operand
        
    Returns:
        Boolean result of equality comparison
    """
    if type(left_operand) == type(right_operand):
        return left_operand == right_operand
    else:
        interpreter._error_manager.report_error("Interpreter: Type mismatch in equality comparison")

def handle_inequality(interpreter, left_operand, right_operand):
    """
    Implements inequality comparison between two values of the same type.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First operand
        right_operand: Second operand
        
    Returns:
        Boolean result of inequality comparison
    """
    if type(left_operand) == type(right_operand):
        return left_operand != right_operand
    else:
        interpreter._error_manager.report_error("Interpreter: Type mismatch in inequality comparison")

def handle_arithmetic(interpreter, left_operand, right_operand, operation):
    """
    Applies arithmetic operations (+, -, *, /, **) on two integer operands.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First integer operand
        right_operand: Second integer operand
        operation: Function implementing the arithmetic operation
        
    Returns:
        Integer result of the arithmetic operation
    """
    if isinstance(left_operand, int) and isinstance(right_operand, int):
        return operation(left_operand, right_operand)
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid types for arithmetic operation")

def handle_concatenation(interpreter, left_operand, right_operand):
    """
    Implements string concatenation operation.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First string operand
        right_operand: Second string operand
        
    Returns:
        Concatenated string
    """
    if isinstance(left_operand, str) and isinstance(right_operand, str):
        return left_operand + right_operand
    else:
        interpreter._error_manager.report_error("Interpreter: Invalid types for string concatenation")

def handle_comparison(interpreter, left_operand, right_operand, comparison_function):
    """
    Implements comparison operations (gr, ge, ls, le) for integers and strings.
    
    Args:
        interpreter: The interpreter instance
        left_operand: First operand (integer or string)
        right_operand: Second operand (integer or string)
        comparison_function: Function implementing the comparison
        
    Returns:
        Boolean result of the comparison
    """
    if (isinstance(left_operand, int) and isinstance(right_operand, int)) or (isinstance(left_operand, str) and isinstance(right_operand, str)):
        return comparison_function(left_operand, right_operand)
    else:
        interpreter._error_manager.report_error("Interpreter: Type mismatch in comparison operation")

#==============================================================================
# SECTION 2: ENVIRONMENT AND VARIABLE MANAGEMENT
#==============================================================================

def lookup_variable(interpreter, var_name):
    """
    Looks up a variable in the current environment and its ancestors.
    
    Args:
        interpreter: The interpreter instance
        var_name: Name of the variable to look up
        
    Returns:
        Tuple containing the variable type and value
    """
    current_env = interpreter.current_env
    while current_env:
        if var_name in current_env._variables:
            return current_env._variables[var_name]
        current_env = current_env.parent
    
    # Variable not found in any environment
    interpreter._error_manager.report_error(f"Interpreter: Variable '{var_name}' not found")

def add_execution_trace(interpreter, rule_id):
    """
    Adds an entry to the execution trace table.
    
    Args:
        interpreter: The interpreter instance
        rule_id: ID of the execution rule being applied
    """
    trace_data = interpreter.execution_trace
    trace_data.append((
        rule_id,
        interpreter.control_stack.get_all_items()[:],
        interpreter.value_stack.get_all_items()[:],
        [interpreter.current_env.index][:]
    ))

def trace_execution_decorator(rule_id):
    """
    Decorator that adds execution tracing to interpreter methods.
    
    Args:
        rule_id: ID of the execution rule being applied
        
    Returns:
        Decorated function with execution tracing
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self._add_execution_trace(rule_id)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def format_tuple_string(element, output):
    """
    Formats a tuple as a string representation.
    
    Args:
        element: The tuple element to format
        output: Current output string
        
    Returns:
        Formatted string representation of the tuple
    """
    if isinstance(element, list):
        output += "("
        for item in element:
            output = format_tuple_string(item, output)
        output = output[:-1] + ")"
    else:
        if isinstance(element.value, list):
            output += "("
            for item in element.value:
                output = format_tuple_string(item, output)
            output = output[:-1] + "),"
        else:
            output += str(element.value) + ","
    return output

def escape_string(string):
    """
    Escapes special characters in a string.
    
    Args:
        string: The string to escape
        
    Returns:
        Escaped string
    """
    return string.encode('unicode_escape').decode()

#==============================================================================
# SECTION 3: DATA STRUCTURES
#==============================================================================

class DataStack:
    """
    Stack data structure for the interpreter.
    """
    def __init__(self):
        """Initialize an empty stack."""
        self.items = []

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0

    def push(self, item):
        """Push an item onto the stack."""
        self.items.append(item)

    def pop(self):
        """Pop the top item from the stack."""
        return self.items.pop()

    def peek(self):
        """Look at the top item without removing it."""
        return self.items[-1]

    def size(self):
        """Get the number of items in the stack."""
        return len(self.items)
    
    def get_all_items(self):
        """Get all items in the stack."""
        return self.items

class Environment:
    """
    Environment for variable bindings with parent-child relationships.
    """
    # Class variable to track environment indices
    index_counter = -1

    # Built-in functions available in the initial environment
    BUILT_IN_FUNCTIONS = [
        "Print", "Isstring", "Isinteger", "Istruthvalue",
        "Istuple", "Isfunction", "Null", "Order", "Stern",
        "Stem", "ItoS", "neg", "not", "Conc"
    ]

    def __init__(self, parent=None):
        """
        Initialize a new environment.
        
        Args:
            parent: Parent environment (None for global environment)
        """
        # Increment the environment index counter
        Environment.index_counter += 1
        self.index = Environment.index_counter
        
        # Initialize the variable dictionary with a default factory
        self._variables = defaultdict(lambda: [None, None])
        
        # Set the parent environment
        self.parent = parent

        # Initialize built-in functions in the global environment
        if self.index == 0:
            self._initialize_built_ins()

    def _initialize_built_ins(self):
        """Initialize built-in functions in the global environment."""
        built_ins = {func: ["inbuilt-functions", None] for func in self.BUILT_IN_FUNCTIONS}
        self._variables.update(built_ins)

    def bind_variable(self, name, type_name, value):
        """
        Bind a variable in this environment.
        
        Args:
            name: Variable name
            type_name: Type of the variable
            value: Value of the variable
        """
        self._variables[name] = [type_name, value]

    def set_parent(self, parent):
        """
        Set the parent environment.
        
        Args:
            parent: Parent environment
        """
        self.parent = parent
        self._variables['__parent__'] = parent._variables if parent else None

    def reset_counter(self):
        """Reset the environment index counter."""
        Environment.index_counter = -1

class ControlStructure(DataStack):
    """
    Control structure for the interpreter.
    """
    def __init__(self, index):
        """
        Initialize a control structure.
        
        Args:
            index: Index of the control structure
        """
        super().__init__()
        self.elements = self.items
        self.index = index

class ControlElement:
    """
    Element in a control structure.
    """
    def __init__(self, type_name, value, bounded_vars=None, control_idx=None, env=None, operator=None):
        """
        Initialize a control element.
        
        Args:
            type_name: Type of the element
            value: Value of the element
            bounded_vars: Bounded variables (for lambda)
            control_idx: Index of the control structure (for lambda)
            env: Environment (for lambda)
            operator: Operator (for binary operations)
        """
        self.type = type_name
        self.value = value
        self.bounded_vars = bounded_vars
        self.control_idx = control_idx
        self.env = env
        self.operator = operator

#==============================================================================
# SECTION 4: TREE LINEARIZATION
#==============================================================================

class TreeLinearizer:
    """
    Linearizes a syntax tree into control structures.
    """
    def __init__(self):
        """Initialize the linearizer."""
        self.control_structures = []
        
    def linearize(self, syntax_tree):
        """
        Linearize a syntax tree into control structures.
        
        Args:
            syntax_tree: Root node of the syntax tree
            
        Returns:
            List of control structures
        """
        self.preorder_traversal(syntax_tree, 0)
        return self.control_structures
    
    def preorder_traversal(self, node, index):
        """
        Perform a preorder traversal of the syntax tree.
        
        Args:
            node: Current node
            index: Index of the current control structure
        """
        # Create a new control structure if needed
        if len(self.control_structures) <= index:
            self.control_structures.append(ControlStructure(index))
            
        # Handle leaf nodes
        if not node.children:
            self.control_structures[index].push(
                ControlElement(self.parse_token(node.data)[0], self.parse_token(node.data)[1])
            )
            return
        
        # Handle lambda nodes
        if node.data == "lambda":
            if node.children[0].data == ",":
                # Multiple parameters
                var_list = []
                for child in node.children[0].children:
                    var_list.append(self.parse_token(child.data)[1])
                self.control_structures[index].push(
                    ControlElement("lambda", "lambda", var_list, len(self.control_structures))
                )
            else:
                # Single parameter
                self.control_structures[index].push(
                    ControlElement("lambda", "lambda", [self.parse_token(node.children[0].data)[1]], len(self.control_structures))
                )
            # Process the lambda body
            self.preorder_traversal(node.children[1], len(self.control_structures))
            
        # Handle tau nodes (tuples)
        elif node.data == "tau":
            self.control_structures[index].push(ControlElement("tau", len(node.children)))
            for child in node.children:
                self.preorder_traversal(child, index)

        # Handle conditional nodes
        elif node.data == "->":
            # Process the true branch
            self.control_structures[index].push(
                ControlElement("delta", "delta", None, len(self.control_structures))
            )
            self.preorder_traversal(node.children[1], len(self.control_structures))
            
            # Process the false branch
            self.control_structures[index].push(
                ControlElement("delta", "delta", None, len(self.control_structures))
            )
            self.preorder_traversal(node.children[2], len(self.control_structures))
            
            # Process the condition
            self.control_structures[index].push(ControlElement("beta", "beta"))
            self.preorder_traversal(node.children[0], index)
        
        # Handle other nodes
        else:
            self.control_structures[index].push(
                ControlElement(self.parse_token(node.data)[0], self.parse_token(node.data)[1])
            )
            
            # Process the first child
            self.preorder_traversal(node.children[0], index)
            
            # Process the second child if it exists
            if len(node.children) > 1:
                self.preorder_traversal(node.children[1], index)
    
    def parse_token(self, token):
        """
        Parse a token into its type and value.
        
        Args:
            token: Token to parse
            
        Returns:
            Tuple of (type, value)
        """
        output = []
        
        # Handle tokens enclosed in angle brackets
        if token[0] == "<":
            # Handle identifiers
            if len(token) > 3 and token[1:3] == "ID":
                if token[4:-1] in ["Conc", "Print", "Stern", "Stem", "Isstring", "Isinteger", 
                                   "Istruthvalue", "Isfunction", "Null", "Istuple", "Order", 
                                   "ItoS", "not", "neg"]:
                    output = [token[4:-1], token[4:-1]]
                else:
                    output = ["ID", token[4:-1]]
            # Handle integers
            elif len(token) > 4 and token[1:4] == "INT":
                output = ["INT", int(token[5:-1])]
            # Handle strings
            elif len(token) > 4 and token[1:4] == "STR":
                output = ["STR", token[6:-2]]
            # Handle boolean true
            elif token[1:-1] == "true":
                output = ["bool", True]
            # Handle boolean false
            elif token[1:-1] == "false":
                output = ["bool", False]
            # Handle nil
            elif token[1:-1] == "nil":
                output = ["nil", None]
            # Handle other tokens
            else:
                output = [token[1:-1], token[1:-1]]
        # Handle tokens not enclosed in angle brackets
        else:
            output = [token, token]
            
        return output

#==============================================================================
# SECTION 5: ERROR HANDLING
#==============================================================================

class ErrorManager:
    """
    Manages error reporting for the interpreter.
    """
    def __init__(self, interpreter):
        """
        Initialize the error manager.
        
        Args:
            interpreter: The interpreter instance
        """
        self.interpreter = interpreter

    def report_error(self, message):
        """
        Report an error and raise an exception.
        
        Args:
            message: Error message
            
        Raises:
            Exception with the error message
        """
        raise Exception(message)

#==============================================================================
# SECTION 6: INTERPRETER IMPLEMENTATION
#==============================================================================

class RPALInterpreter:
    """
    Interpreter for RPAL programs.
    """
    def __init__(self):
        """Initialize the interpreter."""
        # Initialize the error manager
        self._error_manager = ErrorManager(self)

        # Initialize the tree linearizer
        self._tree_linearizer = TreeLinearizer()
        
        # Initialize the global environment
        self.global_env = Environment()

        # Initialize control structures and stacks
        self.control_structures = None
        self.current_env = self.global_env
        self.value_stack = DataStack()
        self.control_stack = DataStack()
        
        # Initialize output queue and execution trace
        self._output_queue = []
        self.execution_trace = []

        # Define supported binary operators
        self.binary_operators = {
            "+", "-", "/", "*", "**", 
            "eq", "ne", "gr", "ge", "le", "ls",
            ">", "<", ">=", "<=", 
            "or", "&", "aug",  
            "Conc"
        }
        
        # Define supported unary operators
        self.unary_operators = {
            "neg", "not",
            "Print", 
            "Isstring", "Isinteger", "Istruthvalue", "Isfunction", "Null", "Istuple",
            "Order", "Stern", "Stem", "ItoS", "$ConcPartial"
        }
        
        # Define built-in functions
        self.built_in_functions = {
            "Print", 
            "Isstring", "Isinteger", "Istruthvalue", "Isfunction", "Null", "Istuple",
            "Order", "Stern", "Stem", "ItoS", "$ConcPartial"
        }

    def setup(self):
        """Set up the interpreter for execution."""
        # Create the global environment marker
        global_env_marker = ControlElement("env_marker", "env_marker", None, None, self.current_env)

        # Push the global environment marker onto both stacks
        self.value_stack.push(global_env_marker)
        self.control_stack.push(global_env_marker)

        # Push elements from the first control structure onto the control stack
        if self.control_structures:
            elements = self.control_structures[0].elements
            for element in elements:
                self.control_stack.push(element)
        else:
            # Handle empty control structures
            self._error_manager.report_error("Interpreter: No control structures to execute")

    def interpret(self, syntax_tree):
        """
        Interpret a syntax tree.
        
        Args:
            syntax_tree: Root node of the syntax tree
        """
        # Linearize the syntax tree
        self.control_structures = self._tree_linearizer.linearize(syntax_tree)
        
        # Set up the interpreter
        self.setup()
        
        # Execute the program
        while not self.control_stack.is_empty():
            # Get the top elements from both stacks
            control_top = self.control_stack.peek()
            stack_top = self.value_stack.peek()

            # Apply the appropriate execution rule based on the control element type
            if control_top.type in ['ID', 'STR', 'INT', 'bool', 'tuple', 'Y*', 'nil', 'dummy']:
                self.apply_rule1()
            elif control_top.type == "lambda":
                self.apply_rule2()
            elif control_top.type == "env_marker":
                self.apply_rule5()
            elif control_top.value in self.binary_operators and self.value_stack.size() >= 2:
                self.apply_rule6()
            elif control_top.value in self.unary_operators and self.value_stack.size() >= 1:
                self.apply_rule7()
            elif control_top.type == "beta" and self.value_stack.size() >= 1:
                self.apply_rule8()
            elif control_top.type == "tau":
                self.apply_rule9()
            elif control_top.type == "gamma" and stack_top.type == "tuple":
                self.apply_rule10()
            elif control_top.type == "gamma" and stack_top.type == "Y*":
                self.apply_rule12()
            elif control_top.type == "gamma" and stack_top.type == "eta":
                self.apply_rule13()
            elif control_top.type == "gamma" and stack_top.type == "lambda":
                if len(stack_top.bounded_vars) > 1:
                    self.apply_rule11()
                else:
                    self.apply_rule4()
            elif control_top.type == "gamma" and stack_top.type == "ConcPartial":
                self.handle_partial_concatenation()
            else:
                self._error_manager.report_error("Interpreter: Invalid control structure")

    @trace_execution_decorator("1")
    def apply_rule1(self):
        """
        Apply execution rule 1: Handle constants and variables.
        
        If the top of the control stack is a constant (STR, INT, etc.), push it onto the value stack.
        If it's a variable (ID), look up its value and push it onto the value stack.
        """
        control_top = self.control_stack.peek()
        
        # Handle constants
        if control_top.type in ['STR', 'INT', 'bool', 'tuple', 'Y*', 'nil', 'dummy']:
            self.value_stack.push(self.control_stack.pop())
        # Handle variables
        else:
            item = self.control_stack.pop()
            var_name = item.value
            var_info = self._lookup_variable(var_name)
            
            # Handle function values
            if var_info[0] == "eta" or var_info[0] == "lambda":
                self.value_stack.push(var_info[1])
            # Handle other values
            else:
                self.value_stack.push(ControlElement(var_info[0], var_info[1]))
        
    @trace_execution_decorator("2") 
    def apply_rule2(self):
        """
        Apply execution rule 2: Handle lambda expressions.
        
        Pop a lambda expression from the control stack, set its environment to the current environment,
        and push it onto the value stack.
        """
        lambda_expr = self.control_stack.pop()
        lambda_expr.env = self.current_env
        self.value_stack.push(lambda_expr)
        
    @trace_execution_decorator("3")
    def apply_rule3(self):
        """Placeholder for rule 3 (unused)."""
        pass
    
    @trace_execution_decorator("4")
    def apply_rule4(self):
        """
        Apply execution rule 4: Handle function application with a single argument.
        
        Pop a lambda expression and an argument from the value stack, create a new environment,
        bind the argument to the parameter in the new environment, and execute the function body.
        """
        # Check for environment limit to prevent infinite recursion
        if self.current_env.index >= 10000:
            self._error_manager.report_error("Interpreter: Environment limit exceeded")
            return

        # Pop the gamma from the control stack
        self.control_stack.pop()
        
        # Pop the lambda and argument from the value stack
        lambda_expr = self.value_stack.pop()
        argument = self.value_stack.pop()
        
        # Create a new environment for the function execution
        new_env = Environment()
        
        # Bind the argument to the parameter
        if argument.type == "eta" or argument.type == "lambda":
            new_env.bind_variable(lambda_expr.bounded_vars[0], argument.type, argument)
        elif argument.type in ["tuple", "INT", "bool", "STR", "nil"]:
            new_env.bind_variable(lambda_expr.bounded_vars[0], argument.type, argument.value)
        else:
            self._error_manager.report_error("Interpreter: Invalid argument type")
            
        # Set the parent environment
        new_env.parent = lambda_expr.env

        # Update the current environment
        self.current_env = new_env
        
        # Create an environment marker
        new_env_marker = ControlElement("env_marker", "env_marker", None, None, new_env)
        
        # Push the environment marker onto the control stack
        self.control_stack.push(new_env_marker) 
        
        # Push the function body onto the control stack
        for element in self.control_structures[lambda_expr.control_idx].elements:
            self.control_stack.push(element)
            
        # Push the environment marker onto the value stack
        self.value_stack.push(new_env_marker)
        
    @trace_execution_decorator("5")
    def apply_rule5(self):
        """
        Apply execution rule 5: Handle environment markers.
        
        Pop an environment marker from the control stack, pop a value and an environment marker
        from the value stack, and push the value back onto the value stack.
        """
        # Pop the environment marker from the control stack
        env = self.control_stack.pop().env
        
        # Pop the value and environment marker from the value stack
        value = self.value_stack.pop()
        stack_env = self.value_stack.pop().env
        
        # Check if the environments match
        if env == stack_env:
            # Push the value back onto the value stack
            self.value_stack.push(value)
            
            # Update the current environment
            for element in reversed(self.value_stack.get_all_items()):
                if element.type == "env_marker":
                    self.current_env = element.env
                    break
        else:
            self._error_manager.report_error("Interpreter: Environment mismatch")
                
    @trace_execution_decorator("6")
    def apply_rule6(self):
        """
        Apply execution rule 6: Handle binary operations.
        
        Pop a binary operator from the control stack, pop two operands from the value stack,
        apply the binary operation, and push the result onto the value stack.
        """
        # Pop the binary operator from the control stack
        binary_op = self.control_stack.pop().value
        
        # Pop the operands from the value stack
        right_operand = self.value_stack.pop()
        left_operand = self.value_stack.pop()
        
        # Handle augmentation
        if binary_op == "aug":
            self.value_stack.push(self._execute_binary_operation(right_operand, left_operand, binary_op))
        # Handle string concatenation
        elif binary_op == "Conc":
            if right_operand.type == "STR" and left_operand.type == "STR":
                result = self._execute_binary_operation(right_operand.value, left_operand.value, binary_op)
                self.value_stack.push(ControlElement("STR", result))
                self.remove_gamma()
                self.remove_gamma()
            elif right_operand.type == "STR":
                right_operand.type = "ConcPartial"
                self.value_stack.push(left_operand)
                self.value_stack.push(right_operand)
                self.remove_gamma()
            else:
                self._error_manager.report_error("Interpreter: Invalid type for string concatenation")
        # Handle other binary operations
        else:
            right_value = right_operand.value
            left_value = left_operand.value
            result = self._execute_binary_operation(right_value, left_value, binary_op)
            
            # Determine the result type
            result_type = "bool" if isinstance(result, bool) else "INT"
            
            # Push the result onto the value stack
            self.value_stack.push(ControlElement(result_type, result))
        
    @trace_execution_decorator("7")    
    def apply_rule7(self):
        """
        Apply execution rule 7: Handle unary operations.
        
        Pop a unary operator from the control stack, pop an operand from the value stack,
        apply the unary operation, and push the result onto the value stack.
        """
        # Pop the unary operator from the control stack
        unary_op = self.control_stack.pop().value
        
        # Pop the operand from the value stack
        operand = self.value_stack.pop()
        
        # Apply the unary operation
        result = self._execute_unary_operation(operand, unary_op)
        
        # Determine the result type
        if isinstance(result, bool):
            result_type = "bool"
        elif isinstance(result, str):
            result_type = "STR"
        else:
            result_type = "INT"
            
        # Remove gamma for built-in functions
        if unary_op in self.built_in_functions:
            self.remove_gamma()
            
        # Push the result onto the value stack
        self.value_stack.push(ControlElement(result_type, result))
                    
    @trace_execution_decorator("8")
    def apply_rule8(self):
        """
        Apply execution rule 8: Handle conditional expressions.
        
        Pop a condition from the value stack, and based on its value, execute either
        the true branch or the false branch.
        """
        # Pop the condition from the value stack
        condition = self.value_stack.pop().value
        
        # Handle true condition
        if condition == True:
            # Pop the beta and first delta from the control stack
            self.control_stack.pop()
            self.control_stack.pop()
            
            # Pop the second delta from the control stack
            delta = self.control_stack.pop()
            
            # Push the true branch onto the control stack
            for element in self.control_structures[delta.control_idx].elements:
                self.control_stack.push(element)
        # Handle false condition
        elif condition == False:
            # Pop the beta from the control stack
            self.control_stack.pop()
            
            # Pop the first delta from the control stack
            delta = self.control_stack.pop()
            
            # Pop the second delta from the control stack
            self.control_stack.pop()
            
            # Push the false branch onto the control stack
            for element in self.control_structures[delta.control_idx].elements:
                self.control_stack.push(element)
        # Handle invalid condition
        else:
            self._error_manager.report_error("Interpreter: Invalid condition type")

    @trace_execution_decorator("9")
    def apply_rule9(self):
        """
        Apply execution rule 9: Handle tuple creation.
        
        Pop a tau from the control stack, pop n elements from the value stack,
        create a tuple with those elements, and push it onto the value stack.
        """
        # Pop the tau from the control stack
        tau = self.control_stack.pop()
        
        # Get the number of elements in the tuple
        n = tau.value
        
        # Pop n elements from the value stack
        tuple_elements = []
        for _ in range(n):
            tuple_elements.append(self.value_stack.pop())
            
        # Push the tuple onto the value stack
        self.value_stack.push(ControlElement("tuple", tuple_elements))

    @trace_execution_decorator("10")   
    def apply_rule10(self):
        """
        Apply execution rule 10: Handle tuple indexing.
        
        Pop a gamma from the control stack, pop a tuple and an index from the value stack,
        get the element at the given index, and push it onto the value stack.
        """
        # Pop the gamma from the control stack
        self.control_stack.pop()
        
        # Pop the tuple and index from the value stack
        tuple_value = self.value_stack.pop()
        index = self.value_stack.pop()
        
        # Check if the index is valid
        if index.type != "INT":
            self._error_manager.report_error("Interpreter: Invalid tuple index type")
            
        # Adjust the index (1-based to 0-based)
        adjusted_index = index.value - 1
        
        # Push the element at the given index onto the value stack
        self.value_stack.push(tuple_value.value[adjusted_index])
                
    @trace_execution_decorator("11")
    def apply_rule11(self):
        """
        Apply execution rule 11: Handle function application with multiple arguments.
        
        Pop a lambda expression and a tuple of arguments from the value stack, create a new environment,
        bind the arguments to the parameters in the new environment, and execute the function body.
        """
        # Pop the gamma from the control stack
        self.control_stack.pop()
        
        # Pop the lambda and arguments from the value stack
        lambda_expr = self.value_stack.pop()
        args_tuple = self.value_stack.pop()
        
        # Get the parameter list, control structure index, and environment
        param_list = lambda_expr.bounded_vars
        control_idx = lambda_expr.control_idx
        lambda_env = lambda_expr.env
        
        # Create a new environment
        new_env = Environment()
        
        # Check if the number of arguments matches the number of parameters
        if len(param_list) != len(args_tuple.value):
            self._error_manager.report_error("Interpreter: Argument count mismatch")
            
        # Bind the arguments to the parameters
        for i in range(len(param_list)):
            arg = args_tuple.value[i]
            if arg.type == "eta" or arg.type == "lambda":
                new_env.bind_variable(param_list[i], arg.type, arg)
            else:
                new_env.bind_variable(param_list[i], arg.type, arg.value)
        
        # Set the parent environment
        new_env.parent = lambda_env
        
        # Update the current environment
        self.current_env = new_env
        
        # Create an environment marker
        env_marker = ControlElement("env_marker", "env_marker", None, None, new_env)
        
        # Push the environment marker onto both stacks
        self.value_stack.push(env_marker)
        self.control_stack.push(env_marker)
        
        # Push the function body onto the control stack
        for element in self.control_structures[control_idx].elements:
            self.control_stack.push(element)

    @trace_execution_decorator("12")       
    def apply_rule12(self):
        """
        Apply execution rule 12: Handle Y* operator (fixed-point combinator).
        
        Pop a gamma from the control stack, pop a Y* and a lambda from the value stack,
        create an eta with the lambda, and push it onto the value stack.
        """
        # Pop the gamma from the control stack
        self.control_stack.pop()
        
        # Pop the Y* and lambda from the value stack
        self.value_stack.pop()  # Y*
        lambda_expr = self.value_stack.pop()
        
        # Check if the popped value is a lambda
        if lambda_expr.type != "lambda":
            self._error_manager.report_error("Interpreter: Expected lambda for Y* operator")
            
        # Create an eta with the lambda
        eta = ControlElement(
            "eta", "eta", 
            lambda_expr.bounded_vars, 
            lambda_expr.control_idx, 
            lambda_expr.env
        )
        
        # Push the eta onto the value stack
        self.value_stack.push(eta)
        
    @trace_execution_decorator("13")
    def apply_rule13(self):
        """
        Apply execution rule 13: Handle eta (delayed application).
        
        Push a gamma onto the control stack, peek at the eta on the value stack,
        and push a lambda with the same properties onto the value stack.
        """
        # Push a gamma onto the control stack
        self.control_stack.push(ControlElement("gamma", "gamma"))
        
        # Peek at the eta on the value stack
        eta = self.value_stack.peek()
        
        # Push a lambda with the same properties onto the value stack
        self.value_stack.push(ControlElement(
            "lambda", "lambda",
            eta.bounded_vars,
            eta.control_idx,
            eta.env
        ))
    
    def handle_partial_concatenation(self):
        """
        Handle partial string concatenation.
        
        Pop a ConcPartial and a string from the value stack, concatenate them,
        and push the result onto the value stack.
        """
        # Pop the operands from the value stack
        right_operand = self.value_stack.pop()
        left_operand = self.value_stack.pop()
        
        # Check if the left operand is a string
        if left_operand.type == "STR":
            # Concatenate the strings
            result = self._execute_binary_operation(right_operand.value, left_operand.value, "Conc")
            
            # Push the result onto the value stack
            self.value_stack.push(ControlElement("STR", result))
            
            # Remove the gamma
            self.remove_gamma()
        else:
            self._error_manager.report_error("Interpreter: Invalid type for string concatenation")

    #--------------------------------------------------------------------------
    # Helper methods
    #--------------------------------------------------------------------------
    
    def _lookup_variable(self, var_name):
        """
        Look up a variable in the current environment and its ancestors.
        
        Args:
            var_name: Name of the variable to look up
            
        Returns:
            Tuple containing the variable type and value
        """
        return lookup_variable(self, var_name)
            
    def _execute_binary_operation(self, left_operand, right_operand, operation):
        """
        Execute a binary operation.
        
        Args:
            left_operand: Left operand
            right_operand: Right operand
            operation: Binary operation to perform
            
        Returns:
            Result of the binary operation
        """
        return execute_binary_operation(self, left_operand, right_operand, operation)
                
    def _execute_unary_operation(self, operand, operation):
        """
        Execute a unary operation.
        
        Args:
            operand: Operand
            operation: Unary operation to perform
            
        Returns:
            Result of the unary operation
        """
        return execute_unary_operation(self, operand, operation)     

    def _add_execution_trace(self, rule_id):
        """
        Add an entry to the execution trace.
        
        Args:
            rule_id: ID of the execution rule being applied
        """
        add_execution_trace(self, rule_id)

    def get_output(self):
        """
        Get the output of the program.
        
        Returns:
            String containing the program output
        """
        return "".join(self._output_queue) + "\n"
    
    def get_escaped_output(self):
        """
        Get the escaped output of the program.
        
        Returns:
            Escaped string containing the program output
        """
        return escape_string(self.get_output())
    
    def remove_gamma(self):
        """Remove a gamma from the control stack if present."""
        if self.control_stack.peek().type == "gamma":
            self.control_stack.pop()

