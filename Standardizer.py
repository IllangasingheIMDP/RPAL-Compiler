from typing import Any
from parser import ASTNode, parse_rpal
from lexical import rpal_lexer

# -----------------------------------------------------------
# STNode: Node in the Standardized Tree (ST) for RPAL
# -----------------------------------------------------------
class STNode:
    """
    Node in the Standardized Tree (ST) for RPAL.
    Each node has a type, an optional value, and a list of children.
    The ST is a normalized version of the AST, suitable for CSE evaluation.
    """
    def __init__(self, node_type: str, value: Any = None):
        self.node_type = node_type    # The type of the node (e.g., 'let', 'gamma', 'lambda', etc.)
        self.value = value            # The value of the node (for leaves like identifiers, integers, strings)
        self.children = []            # List of child STNode objects
        self.depth = 0                # Depth in the tree (for pretty printing)
        self.parent = None            # Parent node (for tree navigation)
        self.is_standardized = False  # Flag to avoid redundant standardization
    
    def set_data(self, data):
        """Set the node type (used during standardization)."""
        self.node_type = data
    
    def get_data(self):
        """Get the node type."""
        return self.node_type
    
    def set_depth(self, depth):
        """Set the depth of the node in the tree."""
        self.depth = depth
    
    def get_depth(self):
        """Get the depth of the node in the tree."""
        return self.depth
    
    def set_parent(self, parent):
        """Set the parent node."""
        self.parent = parent
    
    def get_parent(self):
        """Get the parent node."""
        return self.parent
    
    def add_child(self, child):
        """Add a child node to this node and update its parent and depth."""
        self.children.append(child)
        if isinstance(child, STNode):
            child.set_parent(self)
            child.set_depth(self.depth + 1)
    
    def standardize(self):
        """
        Standardize the current node and all its children recursively.
        Converts AST-like structures to canonical ST forms for CSE evaluation.
        """
        if not self.is_standardized:
            # Standardize all children first (post-order traversal)
            for child in self.children:
                if isinstance(child, STNode):
                    child.standardize()

            # Standardize current node based on its type
            if self.node_type == "let":
                # let X = E in P  ==>  (lambda X.P) E  ==> gamma(lambda(X, P), E)
                temp1 = self.children[0].children[1]  # E
                temp1.set_parent(self)
                temp1.set_depth(self.depth + 1)
                
                temp2 = self.children[1]  # P
                temp2.set_parent(self.children[0])
                temp2.set_depth(self.depth + 2)
                
                self.children[1] = temp1
                self.children[0].set_data("lambda")
                self.children[0].children[1] = temp2
                self.set_data("gamma")

            elif self.node_type == "where":
                # P where X = E  ==>  let X = E in P
                temp = self.children[0]  # P
                self.children[0] = self.children[1]  # X = E
                self.children[1] = temp
                self.set_data("let")
                self.standardize()

            elif self.node_type == "function_form":
                # f x y = E  ==>  f = lambda x.lambda y.E
                Ex = self.children[-1]  # E
                current_lambda = STNode("lambda")
                current_lambda.set_depth(self.depth + 1)
                current_lambda.set_parent(self)
                self.children.insert(1, current_lambda)

                # Create a chain of lambda nodes for each parameter
                i = 2
                while i < len(self.children) - 1:
                    param = self.children[i]
                    current_lambda.add_child(param)
                    if i < len(self.children) - 2:
                        new_lambda = STNode("lambda")
                        new_lambda.set_depth(current_lambda.depth + 1)
                        new_lambda.set_parent(current_lambda)
                        current_lambda.add_child(new_lambda)
                        current_lambda = new_lambda
                    i += 1

                current_lambda.add_child(Ex)
                self.children = self.children[:2]  # Keep only function name and first lambda
                self.set_data("=")

            elif self.node_type == "lambda":
                # Standardize multiple-parameter lambda into a chain of unary lambdas
                if len(self.children) > 2:
                    Ey = self.children[-1]
                    current_lambda = STNode("lambda")
                    current_lambda.set_depth(self.depth + 1)
                    current_lambda.set_parent(self)
                    self.children.insert(1, current_lambda)

                    i = 2
                    while i < len(self.children) - 1:
                        param = self.children[i]
                        current_lambda.add_child(param)
                        if i < len(self.children) - 2:
                            new_lambda = STNode("lambda")
                            new_lambda.set_depth(current_lambda.depth + 1)
                            new_lambda.set_parent(current_lambda)
                            current_lambda.add_child(new_lambda)
                            current_lambda = new_lambda
                        i += 1

                    current_lambda.add_child(Ey)
                    self.children = self.children[:2]

            elif self.node_type == "within":
                # X1 = E1 within X2 = E2  ==>  X2 = (λX1.E2)E1
                X1 = self.children[0].children[0]
                X2 = self.children[1].children[0]
                E1 = self.children[0].children[1]
                E2 = self.children[1].children[1]

                gamma = STNode("gamma")
                lambda_node = STNode("lambda")

                # Set relationships and depths
                gamma.set_depth(self.depth + 1)
                gamma.set_parent(self)
                lambda_node.set_depth(self.depth + 2)
                lambda_node.set_parent(gamma)

                # Adjust depths and parents for children
                X1.set_depth(lambda_node.depth + 1)
                X1.set_parent(lambda_node)
                X2.set_depth(self.depth + 1)
                X2.set_parent(self)
                E1.set_depth(gamma.depth)
                E1.set_parent(gamma)
                E2.set_depth(lambda_node.depth + 1)
                E2.set_parent(lambda_node)

                # Build the structure
                lambda_node.add_child(X1)
                lambda_node.add_child(E2)
                gamma.add_child(lambda_node)
                gamma.add_child(E1)
                
                self.children = [X2, gamma]
                self.set_data("=")

            elif self.node_type == "@":
                # E1 @ N E2  ==>  (N E1) E2  ==> gamma(gamma(N, E1), E2)
                gamma1 = STNode("gamma")
                gamma1.set_depth(self.depth + 1)
                gamma1.set_parent(self)

                e1 = self.children[0]
                n = self.children[1]
                
                e1.set_depth(e1.depth + 1)
                e1.set_parent(gamma1)
                n.set_depth(n.depth + 1)
                n.set_parent(gamma1)

                gamma1.add_child(n)
                gamma1.add_child(e1)
                self.children.pop(0)
                self.children.pop(0)
                self.children.insert(0, gamma1)
                self.set_data("gamma")

            elif self.node_type == "and":
                # and standardization: simultaneous definitions
                # and(X1=E1, X2=E2, ...) ==> (X1, X2, ...) = (E1, E2, ...)
                comma = STNode(",")
                tau = STNode("tau")

                comma.set_depth(self.depth + 1)
                tau.set_depth(self.depth + 1)
                comma.set_parent(self)
                tau.set_parent(self)

                for equal in self.children:
                    equal.children[0].set_parent(comma)
                    equal.children[1].set_parent(tau)
                    comma.add_child(equal.children[0])
                    tau.add_child(equal.children[1])

                self.children = [comma, tau]
                self.set_data("=")

            elif self.node_type == "rec":
                # rec f = E  ==>  f = (Y* (λf.E))
                if len(self.children) > 0 and self.children[0].node_type == "=":
                    X = self.children[0].children[0]
                    E = self.children[0].children[1]
                    
                    # Create new nodes for the transformation
                    F = STNode(X.get_data(), X.value)
                    G = STNode("gamma")
                    Y = STNode("Y*")
                    L = STNode("lambda")
                    
                    # Set depths
                    F.set_depth(self.depth + 1)
                    G.set_depth(self.depth + 1)
                    Y.set_depth(self.depth + 2)
                    L.set_depth(self.depth + 2)
                    
                    # Set parents
                    F.set_parent(self)
                    G.set_parent(self)
                    Y.set_parent(G)
                    L.set_parent(G)
                    
                    # Build structure for recursion
                    X.set_depth(L.depth + 1)
                    X.set_parent(L)
                    E.set_depth(L.depth + 1)
                    E.set_parent(L)
                    
                    L.add_child(X)
                    L.add_child(E)
                    G.add_child(Y)
                    G.add_child(L)
                    
                    self.children = [F, G]
                    self.set_data("=")

            self.is_standardized = True
    
    def print_tree(self, indent=0):
        """
        Print the ST tree with indentation and RPAL-style tags.
        - Node types in no_bracket_nodes are printed without angle brackets.
        - <->> is printed as ->.
        - <Ystar> is printed as <Y*>.
        - Strings are printed as <STR:'value'> (with quotes).
        """
        no_bracket_nodes = {"tau", "ls", "gr", "rec", "gamma", "lambda", "+", "->", "&", ",", "-", "eq"}
        # Special case: print <->> as ->
        if self.node_type == "->>":
            print("." * indent + "->")
        elif self.node_type == "Y*":
            print("." * indent + "<Y*>")
        elif self.value is not None:
            if self.node_type in ["ID", "identifier"]:
                print("." * indent + f"<ID:{self.value}>")
            elif self.node_type in ["INT", "integer"]:
                print("." * indent + f"<INT:{self.value}>")
            elif self.node_type in ["STR", "string"]:
                print("." * indent + f"<STR:'{self.value}'>")
            elif self.node_type in no_bracket_nodes:
                print("." * indent + f"{self.node_type}")
            else:
                print("." * indent + f"<{self.node_type}:{self.value}>")
        else:
            if self.node_type in no_bracket_nodes:
                print("." * indent + f"{self.node_type}")
            else:
                print("." * indent + f"<{self.node_type}>")
        
        for child in self.children:
            if isinstance(child, STNode):
                child.print_tree(indent + 1)
            else:
                print("." * (indent + 1) + str(child))

    def __str__(self):
        """String representation of the node for debugging."""
        if self.value is not None:
            return f"<{self.node_type}:{self.value}>"
        return f"<{self.node_type}>"
    
# -----------------------------------------------------------
# AST to ST Conversion Utilities
# -----------------------------------------------------------
def ast_to_st(ast_node: ASTNode) -> STNode:
    """
    Convert an AST (Abstract Syntax Tree) node to a Standardized Tree (ST) node.
    Standardizes the tree after conversion.
    """
    st = convert_ast_to_st(ast_node)
    st.standardize()
    return st

def convert_ast_to_st(ast_node: ASTNode) -> STNode:
    """
    Recursively convert AST nodes to ST nodes.
    """
    st_node = STNode(ast_node.node_type, ast_node.value)
    for child in ast_node.children:
        st_child = convert_ast_to_st(child)
        st_node.add_child(st_child)
    return st_node

def standardize_program(input_string: str) -> STNode:
    """
    Parse and standardize an RPAL program from source code.
    Returns the root of the Standardized Tree (ST).
    """
    ast = parse_rpal(input_string)
    return ast_to_st(ast)

# -----------------------------------------------------------
# Testing: Run this file directly to test the standardizer
# -----------------------------------------------------------
if __name__ == '__main__':
    # Test programs for standardization
    test_programs = [
        "let rec Rev S = S eq '' -> '' | (Rev(Stern S)) @Conc (Stem S ) in let Pairs (S1,S2) = P (Rev S1, Rev S2) where rec P (S1, S2) = S1 eq '' & S2 eq '' -> nil |  (fn L. P (Stern S1, Stern S2) aug ((Stem S1) @Conc (Stem S2))) nil in Print ( Pairs ('abc','def'))"
    ]
    
    print("RPAL Standardizer Test\n")
    
    for i, program in enumerate(test_programs, 1):
        print(f"\nTest Program {i}:")
        print("="*80)
        try:
            st = standardize_program(program)
            print("\nStandardizing AST:")
            print("="*80)
            print(st.print_tree(0))
            print("="*80)
            print("\nStandardized Tree:")
            print("="*80)
            print(f"\nStandardization successful!")
        except Exception as e:
            print(f"Error standardizing program: {e}")
        print("="*80)
        print()
