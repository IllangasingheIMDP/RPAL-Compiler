class BooleanParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos][0] if self.tokens else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos][0]
        else:
            self.current_token = None

    def match(self, expected_token):
        if self.current_token == expected_token:
            self.advance()
        else:
            raise SyntaxError(f"Expected {expected_token}, got {self.current_token}")

    def parse(self):
        self.E()
        if self.current_token is not None:
            raise SyntaxError("Extra tokens after valid expression")
        print("Parsing completed successfully.")

    # E -> T X
    def E(self):
        self.T()
        self.X()

    # X -> or T X | nor T X | xor T X | ε
    def X(self):
        while self.current_token in ['OR_OP', 'NOR_OP', 'XOR_OP']:
            self.advance()
            self.T()

    # T -> F Y
    def T(self):
        self.F()
        self.Y()

    # Y -> and F Y | nand F Y | ε
    def Y(self):
        while self.current_token in ['AND_OP', 'NAND_OP']:
            self.advance()
            self.F()

    # F -> not F | P
    def F(self):
        if self.current_token == 'NOT_OP':
            self.advance()
            self.F()
        else:
            self.P()

    # P -> ( E ) | i | true | false
    def P(self):
        if self.current_token == 'LPAREN':
            self.match('LPAREN')
            self.E()
            self.match('RPAREN')
        elif self.current_token == 'IDENTIFIER':
            self.match('IDENTIFIER')
        elif self.current_token == 'TRUE':
            self.match('TRUE')
        elif self.current_token == 'FALSE':
            self.match('FALSE')
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")


if __name__ == '__main__':
    input_string = 'true nand (false xor i) or not i and not false nor i'
    
    # Call lexer
    token_list = lexer(input_string)

    # Call parser
    parser = BooleanParser(token_list)
    parser.parse()
