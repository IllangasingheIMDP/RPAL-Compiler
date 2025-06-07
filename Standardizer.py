from typing import Any
from parser import ASTNode, parse_rpal
from lexical import rpal_lexer

class STNode:
    """
    Node in the Standardized Tree (ST) for RPAL
    """
    def __init__(self, node_type: str, value: Any = None):
        self.node_type = node_type
        self.value = value
        self.children = []
        self.depth = 0
        self.parent = None
        self.is_standardized = False
    
    def set_data(self, data):
        self.node_type = data
    
    def get_data(self):
        return self.node_type
    
    def set_depth(self, depth):
        self.depth = depth
    
    def get_depth(self):
        return self.depth
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_parent(self):
        return self.parent
    
    def add_child(self, child):
        """Add a child node to this node"""
        self.children.append(child)
        if isinstance(child, STNode):
            child.set_parent(self)
            child.set_depth(self.depth + 1)
    
    def standardize(self):
        if not self.is_standardized:
            # Standardize all children first
            for child in self.children:
                if isinstance(child, STNode):
                    child.standardize()

            # Standardize current node based on type
            if self.node_type == "let":
                # Convert let to gamma/lambda structure
                # let X = E in P -> (lambda X.P) E
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
                # Convert where to let
                # P where X = E -> let X = E in P
                temp = self.children[0]  # P
                self.children[0] = self.children[1]  # X = E
                self.children[1] = temp
                self.set_data("let")
                self.standardize()

            elif self.node_type == "function_form":
                # Convert function form to lambda chain
                # f x y = E -> f = lambda x.lambda y.E
                Ex = self.children[-1]  # E
                current_lambda = STNode("lambda")
                current_lambda.set_depth(self.depth + 1)
                current_lambda.set_parent(self)
                self.children.insert(1, current_lambda)

                # Create lambda chain
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
                # Lambda chain standardization
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
                # within standardization: (X1 = E1 within X2 = E2) -> X2 = (λX1.E2)E1
                X1 = self.children[0].children[0]
                X2 = self.children[1].children[0]
                E1 = self.children[0].children[1]
                E2 = self.children[1].children[1]

                gamma = STNode("gamma")
                lambda_node = STNode("lambda")

                # Set relationships
                gamma.set_depth(self.depth + 1)
                gamma.set_parent(self)
                lambda_node.set_depth(self.depth + 2)
                lambda_node.set_parent(gamma)

                # Adjust depths and parents
                X1.set_depth(lambda_node.depth + 1)
                X1.set_parent(lambda_node)
                X2.set_depth(self.depth + 1)
                X2.set_parent(self)
                E1.set_depth(gamma.depth)
                E1.set_parent(gamma)
                E2.set_depth(lambda_node.depth + 1)
                E2.set_parent(lambda_node)

                # Build structure
                lambda_node.add_child(X1)
                lambda_node.add_child(E2)
                gamma.add_child(lambda_node)
                gamma.add_child(E1)
                
                self.children = [X2, gamma]
                self.set_data("=")

            elif self.node_type == "@":
                # @ standardization: E1@N E2 -> (N E1)E2
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
                # and standardization: simultaneous definition
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
                # rec standardization: rec f = E -> f = (Y* (λf.E))
                if len(self.children) > 0 and self.children[0].node_type == "=":
                    X = self.children[0].children[0]
                    E = self.children[0].children[1]
                    
                    # Create new nodes
                    F = STNode(X.get_data(), X.value)
                    G = STNode("gamma")
                    Y = STNode("Ystar")
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
                    
                    # Build structure
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
        """Print the ST tree with indentation and RPAL-style tags"""
        no_bracket_nodes = {"tau", "ls", "gr", "rec", "gamma", "lambda", "+", "->", "&", ",", "-", "eq"}
        # Special case: print <->> as ->
        if self.node_type == "->>":
            print("." * indent + "->")
        elif self.node_type == "Ystar":
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
        """String representation of the node"""
        if self.value is not None:
            return f"<{self.node_type}:{self.value}>"
        return f"<{self.node_type}>"
    
def ast_to_st(ast_node: ASTNode) -> STNode:
    """Convert AST to Standardized Tree"""
    st = convert_ast_to_st(ast_node)
    st.standardize()
    return st

def convert_ast_to_st(ast_node: ASTNode) -> STNode:
    """Helper function to convert AST nodes to ST nodes"""
    st_node = STNode(ast_node.node_type, ast_node.value)
    for child in ast_node.children:
        st_child = convert_ast_to_st(child)
        st_node.add_child(st_child)
    return st_node

def standardize_program(input_string: str) -> STNode:
    """Parse and standardize RPAL program"""
    ast = parse_rpal(input_string)
    return ast_to_st(ast)

if __name__ == '__main__':
    # Test programs
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
