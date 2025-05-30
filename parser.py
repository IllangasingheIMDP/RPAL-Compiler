import sys
from typing import List, Optional, Union, Any, Tuple
from lexical import Token, TokenType, rpal_lexer

class ASTNode:
    """
    Node in the Abstract Syntax Tree for RPAL
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
            return f"{self.node_type}({self.value})"
        return self.node_type
    
    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent=0):
        """Print the tree with indentation"""
        if self.value is not None:
            if self.node_type == "identifier":
                print("." * indent + f"<ID:{self.value}>")
            elif self.node_type == "integer":
                print("." * indent + f"<INT:{self.value}>")
            elif self.node_type == "string":
                print("." * indent + f"<STR:{self.value}>")
            else:
                print("." * indent + f"<{self.node_type}:{self.value}>")
        else:
            print("." * indent + f"<{self.node_type}>")
        
        for child in self.children:
            child.print_tree(indent + 1)


class Parser:
    """
    Recursive Descent Parser for RPAL language
    """
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current_token_index = 0
        self.current_token = self.tokens[0]
    
    def parse(self) -> ASTNode:
        """Parse the tokens and return the AST root node"""
        return self.parse_E()
    
    def advance(self):
        """Move to the next token"""
        self.current_token_index += 1
        if self.current_token_index < len(self.tokens):
            self.current_token = self.tokens[self.current_token_index]
    
    def match(self, token_type: TokenType) -> Token:
        """Match the current token with the expected type and advance"""
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            raise SyntaxError(f"Expected {token_type.value}, got {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
    
    def peek(self) -> Token:
        """Look at the current token without advancing"""
        return self.current_token
    
    def check(self, token_type: TokenType) -> bool:
        """Check if the current token is of the expected type"""
        return self.current_token.type == token_type
    
    def check_any(self, token_types: List[TokenType]) -> bool:
        """Check if the current token is any of the expected types"""
        return any(self.check(token_type) for token_type in token_types)
    
    # Grammar rule implementations
    
    # E -> 'let' D 'in' E => 'let'
    #    -> 'fn' Vb+ '.' E => 'lambda'
    #    -> Ew
    def parse_E(self) -> ASTNode:
        if self.check(TokenType.LET):
            self.match(TokenType.LET)
            d_node = self.parse_D()
            self.match(TokenType.IN)
            e_node = self.parse_E()
            
            let_node = ASTNode("let")
            let_node.add_child(d_node)
            let_node.add_child(e_node)
            return let_node
        
        elif self.check(TokenType.FN):
            self.match(TokenType.FN)
            
            # Parse one or more Vb
            vb_nodes = []
            vb_nodes.append(self.parse_Vb())
            
            while not self.check(TokenType.EQUALS) and not self.check(TokenType.PERIOD):
                vb_nodes.append(self.parse_Vb())
            
            # Match the period
            if self.check(TokenType.PERIOD):
                self.match(TokenType.PERIOD)
            else:
                # In RPAL, '.' can be omitted if '=' follows
                pass
            
            e_node = self.parse_E()
            
            # Create lambda node
            lambda_node = ASTNode("lambda")
            
            # Add all variable bindings
            for vb in vb_nodes:
                lambda_node.add_child(vb)
            
            # Add the expression
            lambda_node.add_child(e_node)
            
            return lambda_node
        
        else:
            return self.parse_Ew()
    
    # Ew -> T 'where' Dr => 'where'
    #     -> T
    def parse_Ew(self) -> ASTNode:
        t_node = self.parse_T()
        
        if self.check(TokenType.WHERE):
            self.match(TokenType.WHERE)
            dr_node = self.parse_Dr()
            
            where_node = ASTNode("where")
            where_node.add_child(t_node)
            where_node.add_child(dr_node)
            return where_node
        
        return t_node
    
    # T -> Ta ( ',' Ta )+ => 'tau'
    #    -> Ta
    def parse_T(self) -> ASTNode:
        ta_node = self.parse_Ta()
        
        if self.check(TokenType.COMMA):
            # Create tau node for tuple
            tau_node = ASTNode("tau")
            tau_node.add_child(ta_node)
            
            while self.check(TokenType.COMMA):
                self.match(TokenType.COMMA)
                ta_node = self.parse_Ta()
                tau_node.add_child(ta_node)
            
            return tau_node
        
        return ta_node
    
    # Ta -> Ta 'aug' Tc => 'aug'
    #     -> Tc
    def parse_Ta(self) -> ASTNode:
        tc_node = self.parse_Tc()
        
        while self.check(TokenType.AUG):
            self.match(TokenType.AUG)
            tc_right_node = self.parse_Tc()
            
            aug_node = ASTNode("aug")
            aug_node.add_child(tc_node)
            aug_node.add_child(tc_right_node)
            tc_node = aug_node
        
        return tc_node
    
    # Tc -> B '->' Tc '|' Tc => '->'
    #     -> B
    def parse_Tc(self) -> ASTNode:
        b_node = self.parse_B()
        
        if self.check(TokenType.CONDITIONAL):
            self.match(TokenType.CONDITIONAL)
            tc_then_node = self.parse_Tc()
            self.match(TokenType.BAR)
            tc_else_node = self.parse_Tc()
            
            conditional_node = ASTNode("->")
            conditional_node.add_child(b_node)
            conditional_node.add_child(tc_then_node)
            conditional_node.add_child(tc_else_node)
            return conditional_node
        
        return b_node
    
    # B -> B 'or' Bt => 'or'
    #    -> Bt
    def parse_B(self) -> ASTNode:
        bt_node = self.parse_Bt()
        
        while self.check(TokenType.OR):
            self.match(TokenType.OR)
            bt_right_node = self.parse_Bt()
            
            or_node = ASTNode("or")
            or_node.add_child(bt_node)
            or_node.add_child(bt_right_node)
            bt_node = or_node
        
        return bt_node
    
    # Bt -> Bt '&' Bs => '&'
    #     -> Bs
    def parse_Bt(self) -> ASTNode:
        bs_node = self.parse_Bs()
        
        while self.check(TokenType.AMPERSAND):
            self.match(TokenType.AMPERSAND)
            bs_right_node = self.parse_Bs()
            
            and_node = ASTNode("&")
            and_node.add_child(bs_node)
            and_node.add_child(bs_right_node)
            bs_node = and_node
        
        return bs_node
    
    # Bs -> 'not' Bp => 'not'
    #     -> Bp
    def parse_Bs(self) -> ASTNode:
        if self.check(TokenType.NOT):
            self.match(TokenType.NOT)
            bp_node = self.parse_Bp()
            
            not_node = ASTNode("not")
            not_node.add_child(bp_node)
            return not_node
        
        return self.parse_Bp()
    
    # Bp -> A ('gr' | '>' ) A => 'gr'
    #     -> A ('ge' | '>=') A => 'ge'
    #     -> A ('ls' | '<' ) A => 'ls'
    #     -> A ('le' | '<=') A => 'le'
    #     -> A 'eq' A => 'eq'
    #     -> A 'ne' A => 'ne'
    #     -> A
    def parse_Bp(self) -> ASTNode:
        a_left_node = self.parse_A()
        
        if self.check(TokenType.GR) or self.check(TokenType.GREATER):
            op_type = "gr"
            if self.check(TokenType.GR):
                self.match(TokenType.GR)
            else:
                self.match(TokenType.GREATER)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        elif self.check(TokenType.GE) or self.check(TokenType.GREATER_EQ):
            op_type = "ge"
            if self.check(TokenType.GE):
                self.match(TokenType.GE)
            else:
                self.match(TokenType.GREATER_EQ)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        elif self.check(TokenType.LS) or self.check(TokenType.LESS):
            op_type = "ls"
            if self.check(TokenType.LS):
                self.match(TokenType.LS)
            else:
                self.match(TokenType.LESS)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        elif self.check(TokenType.LE) or self.check(TokenType.LESS_EQ):
            op_type = "le"
            if self.check(TokenType.LE):
                self.match(TokenType.LE)
            else:
                self.match(TokenType.LESS_EQ)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        elif self.check(TokenType.EQ) or self.check(TokenType.EQUALS):
            op_type = "eq"
            if self.check(TokenType.EQ):
                self.match(TokenType.EQ)
            else:
                self.match(TokenType.EQUALS)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        elif self.check(TokenType.NE) or self.check(TokenType.NOT_EQUALS):
            op_type = "ne"
            if self.check(TokenType.NE):
                self.match(TokenType.NE)
            else:
                self.match(TokenType.NOT_EQUALS)
            a_right_node = self.parse_A()
            
            op_node = ASTNode(op_type)
            op_node.add_child(a_left_node)
            op_node.add_child(a_right_node)
            return op_node
        
        return a_left_node
    
    # A -> A '+' At => '+'
    #    -> A '-' At => '-'
    #    -> '+' At
    #    -> '-' At => 'neg'
    #    -> At
    def parse_A(self) -> ASTNode:
        # Handle unary + and -
        if self.check(TokenType.PLUS):
            self.match(TokenType.PLUS)
            at_node = self.parse_At()
            # Unary + is a no-op, just return the operand
            return at_node
        
        elif self.check(TokenType.MINUS):
            self.match(TokenType.MINUS)
            at_node = self.parse_At()
            
            neg_node = ASTNode("neg")
            neg_node.add_child(at_node)
            return neg_node
        
        # Handle binary + and -
        at_node = self.parse_At()
        
        while self.check(TokenType.PLUS) or self.check(TokenType.MINUS):
            if self.check(TokenType.PLUS):
                self.match(TokenType.PLUS)
                at_right_node = self.parse_At()
                
                plus_node = ASTNode("+")
                plus_node.add_child(at_node)
                plus_node.add_child(at_right_node)
                at_node = plus_node
            
            elif self.check(TokenType.MINUS):
                self.match(TokenType.MINUS)
                at_right_node = self.parse_At()
                
                minus_node = ASTNode("-")
                minus_node.add_child(at_node)
                minus_node.add_child(at_right_node)
                at_node = minus_node
        
        return at_node
    
    # At -> At '*' Af => '*'
    #     -> At '/' Af => '/'
    #     -> Af
    def parse_At(self) -> ASTNode:
        af_node = self.parse_Af()
        
        while self.check(TokenType.MULTIPLY) or self.check(TokenType.DIVIDE):
            if self.check(TokenType.MULTIPLY):
                self.match(TokenType.MULTIPLY)
                af_right_node = self.parse_Af()
                
                mult_node = ASTNode("*")
                mult_node.add_child(af_node)
                mult_node.add_child(af_right_node)
                af_node = mult_node
            
            elif self.check(TokenType.DIVIDE):
                self.match(TokenType.DIVIDE)
                af_right_node = self.parse_Af()
                
                div_node = ASTNode("/")
                div_node.add_child(af_node)
                div_node.add_child(af_right_node)
                af_node = div_node
        
        return af_node
    
    # Af -> Ap '**' Af => '**'
    #     -> Ap
    def parse_Af(self) -> ASTNode:
        ap_node = self.parse_Ap()
        
        if self.check(TokenType.POWER):
            self.match(TokenType.POWER)
            af_node = self.parse_Af()
            
            power_node = ASTNode("**")
            power_node.add_child(ap_node)
            power_node.add_child(af_node)
            return power_node
        
        return ap_node
    
    # Ap -> Ap '@' '<IDENTIFIER>' R => '@'
    #     -> R
    def parse_Ap(self) -> ASTNode:
        r_node = self.parse_R()
        
        while self.check(TokenType.AT):
            self.match(TokenType.AT)
            
            if not self.check(TokenType.IDENTIFIER):
                raise SyntaxError(f"Expected IDENTIFIER after '@', got {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
            
            identifier = self.match(TokenType.IDENTIFIER)
            r_right_node = self.parse_R()
            
            at_node = ASTNode("@")
            at_node.add_child(r_node)
            id_node = ASTNode("identifier", identifier.value)
            at_node.add_child(id_node)
            at_node.add_child(r_right_node)
            r_node = at_node
        
        return r_node
    
    # R -> R Rn => 'gamma'
    #    -> Rn
    def parse_R(self) -> ASTNode:
        rn_node = self.parse_Rn()
        
        # Function application (gamma)
        while (self.check(TokenType.IDENTIFIER) or 
               self.check(TokenType.INTEGER) or 
               self.check(TokenType.STRING) or 
               self.check(TokenType.TRUE) or 
               self.check(TokenType.FALSE) or 
               self.check(TokenType.NIL) or 
               self.check(TokenType.LPAREN) or 
               self.check(TokenType.DUMMY) or
               self.check(TokenType.STERN) or    # Added this line
               self.check(TokenType.STEM)): 
            
            
            rn_right_node = self.parse_Rn()
            
            gamma_node = ASTNode("gamma")
            gamma_node.add_child(rn_node)
            gamma_node.add_child(rn_right_node)
            rn_node = gamma_node
        
        return rn_node
    
    # Rn -> '<IDENTIFIER>'
    #     -> '<INTEGER>'
    #     -> '<STRING>'
    #     -> 'true' => 'true'
    #     -> 'false' => 'false'
    #     -> 'nil' => 'nil'
    #     -> '(' E ')'
    #     -> 'dummy' => 'dummy'
    def parse_Rn(self) -> ASTNode:
        if self.check(TokenType.IDENTIFIER):
            identifier = self.match(TokenType.IDENTIFIER)
            return ASTNode("identifier", identifier.value)
        
        elif self.check(TokenType.INTEGER):
            integer = self.match(TokenType.INTEGER)
            return ASTNode("integer", int(integer.value))
        
        elif self.check(TokenType.STRING):
            string = self.match(TokenType.STRING)
            # Remove the surrounding quotes
            string_value = string.value[1:-1]
            return ASTNode("string", string_value)
        
        elif self.check(TokenType.TRUE):
            self.match(TokenType.TRUE)
            return ASTNode("true")
        
        elif self.check(TokenType.FALSE):
            self.match(TokenType.FALSE)
            return ASTNode("false")
        
        elif self.check(TokenType.NIL):
            self.match(TokenType.NIL)
            return ASTNode("nil")
        
        elif self.check(TokenType.LPAREN):
            self.match(TokenType.LPAREN)
            e_node = self.parse_E()
            self.match(TokenType.RPAREN)
            return e_node
        
        elif self.check(TokenType.DUMMY):
            self.match(TokenType.DUMMY)
            return ASTNode("dummy")
        
        elif self.check(TokenType.STERN):
            stern = self.match(TokenType.STERN)
            return ASTNode("identifier", stern.value)

        elif self.check(TokenType.STEM):
            stem = self.match(TokenType.STEM)
            return ASTNode("identifier", stem.value)
        else:
            raise SyntaxError(f"Unexpected token {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
    
    # D -> Da 'within' D => 'within'
    #    -> Da
    def parse_D(self) -> ASTNode:
        da_node = self.parse_Da()
        
        if self.check(TokenType.WITHIN):
            self.match(TokenType.WITHIN)
            d_node = self.parse_D()
            
            within_node = ASTNode("within")
            within_node.add_child(da_node)
            within_node.add_child(d_node)
            return within_node
        
        return da_node
    
    # Da -> Dr ( 'and' Dr )+ => 'and'
    #     -> Dr
    def parse_Da(self) -> ASTNode:
        dr_node = self.parse_Dr()
        
        if self.check(TokenType.AND) or self.check(TokenType.SIMULTDEF):
            # Create and node for simultaneous definitions
            and_node = ASTNode("and")
            and_node.add_child(dr_node)
            
            while self.check(TokenType.AND) or self.check(TokenType.SIMULTDEF):
                if self.check(TokenType.AND):
                    self.match(TokenType.AND)
                else:
                    self.match(TokenType.SIMULTDEF)
                
                dr_right_node = self.parse_Dr()
                and_node.add_child(dr_right_node)
            
            return and_node
        
        return dr_node
    
    # Dr -> 'rec' Db => 'rec'
    #     -> Db
    def parse_Dr(self) -> ASTNode:
        if self.check(TokenType.REC):
            self.match(TokenType.REC)
            db_node = self.parse_Db()
            
            rec_node = ASTNode("rec")
            rec_node.add_child(db_node)
            return rec_node
        
        return self.parse_Db()
    
    # Db -> Vl '=' E => '='
    #     -> '<IDENTIFIER>' Vb+ '=' E => 'fcn_form'
    #     -> '(' D ')'
    def parse_Db(self) -> ASTNode:
        if self.check(TokenType.LPAREN):
            self.match(TokenType.LPAREN)
            d_node = self.parse_D()
            self.match(TokenType.RPAREN)
            return d_node
        
        # Check for function form: <IDENTIFIER> Vb+ '=' E
        elif self.check(TokenType.IDENTIFIER):
            identifier_token = self.peek()
            self.advance()
            
            # Look ahead to see if this is a function form
            if (self.check(TokenType.IDENTIFIER) or 
                self.check(TokenType.LPAREN)):
                
                # This is a function form
                identifier_node = ASTNode("identifier", identifier_token.value)
                
                # Parse one or more Vb
                vb_nodes = []
                vb_nodes.append(self.parse_Vb())
                
                while not self.check(TokenType.EQUALS):
                    vb_nodes.append(self.parse_Vb())
                
                self.match(TokenType.EQUALS)
                e_node = self.parse_E()
                
                # Create function form node
                fcn_form_node = ASTNode("function_form")
                fcn_form_node.add_child(identifier_node)
                
                # Add all variable bindings
                for vb in vb_nodes:
                    fcn_form_node.add_child(vb)
                
                # Add the expression
                fcn_form_node.add_child(e_node)
                
                return fcn_form_node
            
            # Otherwise, it's a simple variable binding
            else:
                # Go back to the identifier
                self.current_token_index -= 1
                self.current_token = self.tokens[self.current_token_index]
                
                vl_node = self.parse_Vl()
                self.match(TokenType.EQUALS)
                e_node = self.parse_E()
                
                equals_node = ASTNode("=")
                equals_node.add_child(vl_node)
                equals_node.add_child(e_node)
                return equals_node
        
        # Handle other variable bindings
        else:
            vl_node = self.parse_Vl()
            self.match(TokenType.EQUALS)
            e_node = self.parse_E()
            
            equals_node = ASTNode("=")
            equals_node.add_child(vl_node)
            equals_node.add_child(e_node)
            return equals_node
    
    # Vb -> '<IDENTIFIER>'
    #     -> '(' Vl ')'
    #     -> '(' ')' => '()'
    def parse_Vb(self) -> ASTNode:
        if self.check(TokenType.IDENTIFIER):
            identifier = self.match(TokenType.IDENTIFIER)
            return ASTNode("identifier", identifier.value)
        
        elif self.check(TokenType.LPAREN):
            self.match(TokenType.LPAREN)
            
            # Check for empty tuple
            if self.check(TokenType.RPAREN):
                self.match(TokenType.RPAREN)
                return ASTNode("()")
            
            vl_node = self.parse_Vl()
            self.match(TokenType.RPAREN)
            return vl_node
        
        else:
            raise SyntaxError(f"Expected IDENTIFIER or '(' in variable binding, got {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
    
    # Vl -> '<IDENTIFIER>' list ',' => ','?
    def parse_Vl(self) -> ASTNode:
        if not self.check(TokenType.IDENTIFIER):
            raise SyntaxError(f"Expected IDENTIFIER in variable list, got {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
        
        identifier = self.match(TokenType.IDENTIFIER)
        id_node = ASTNode("identifier", identifier.value)
        
        # Check for comma-separated list
        if self.check(TokenType.COMMA):
            comma_node = ASTNode(",")
            comma_node.add_child(id_node)
            
            while self.check(TokenType.COMMA):
                self.match(TokenType.COMMA)
                
                if not self.check(TokenType.IDENTIFIER):
                    raise SyntaxError(f"Expected IDENTIFIER after ',' in variable list, got {self.current_token.type.value} at line {self.current_token.line}, column {self.current_token.column}")
                
                next_identifier = self.match(TokenType.IDENTIFIER)
                next_id_node = ASTNode("identifier", next_identifier.value)
                comma_node.add_child(next_id_node)
            
            return comma_node
        
        return id_node


def parse_rpal(input_string: str) -> ASTNode:
    """
    Parse RPAL source code and return the AST
    """
    tokens = rpal_lexer(input_string)
    parser = Parser(tokens)
    return parser.parse()


# ------------------------ Testing ------------------------

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
                ast = parse_rpal(source)
                if ast_only:
                    ast.print_tree(0)
                else:
                    # Import and use StAndCSE for evaluation
                    from StAndCSE import evaluate_rpal
                    evaluate_rpal(source, ast_only=False)
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Test RPAL programs
        test_programs = [
            # Simple let expression
            "let rec Rev S = S eq '' -> '' | (Rev(Stern S)) @Conc (Stem S ) in let Pairs (S1,S2) = P (Rev S1, Rev S2) where rec P (S1, S2) = S1 eq '' & S2 eq '' -> nil |  (fn L. P (Stern S1, Stern S2) aug ((Stem S1) @Conc (Stem S2))) nil in Print ( Pairs ('abc','def'))"
        ]
        
        for i, program in enumerate(test_programs, 1):
            print(f"\n🔍 Test Program {i}: {program}")
            print("=" * 50)
            
            try:
                ast = parse_rpal(program)
                print("AST Structure:")
                ast.print_tree(0)
                    
            except Exception as e:
                print(f"Error: {e}")
            
            print()
