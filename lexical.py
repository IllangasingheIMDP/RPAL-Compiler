import re
from enum import Enum
from typing import List, Tuple, NamedTuple

class TokenType(Enum):
    # Literals
    IDENTIFIER = 'IDENTIFIER'
    INTEGER = 'INTEGER'
    STRING = 'STRING'
    
    # Keywords
    LET = 'LET'
    IN = 'IN'
    WHERE = 'WHERE'
    FN = 'FN'
    REC = 'REC'
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    GR = 'GR'
    GE = 'GE'
    LS = 'LS'
    LE = 'LE'
    EQ = 'EQ'
    NE = 'NE'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    NIL = 'NIL'
    DUMMY = 'DUMMY'
    WITHIN = 'WITHIN'
    SIMULTDEF = 'SIMULTDEF'  # 'and' in simultaneous definitions
    AUG = 'AUG'
    CONC = 'CONC'
    STERN = 'STERN'
    STEM = 'STEM'
    ISTUPLE = 'ISTUPLE'
    ISINTEGER = 'ISINTEGER'
    ISSTRING = 'ISSTRING'
    ISFUNCTION = 'ISFUNCTION'
    ISTRUTHVALUE = 'ISTRUTHVALUE'
    ISDUMMY = 'ISDUMMY'
    ITO = 'ITO'
    ORDER = 'ORDER'
    NULL = 'NULL'
    PRINT = 'PRINT'
    
    # Operators
    PLUS = 'PLUS'           # +
    MINUS = 'MINUS'         # -
    MULTIPLY = 'MULTIPLY'   # *
    DIVIDE = 'DIVIDE'       # /
    POWER = 'POWER'         # **
    AT = 'AT'               # @
    
    # Comparison operators
    LESS = 'LESS'           # <
    LESS_EQ = 'LESS_EQ'     # <=
    GREATER = 'GREATER'     # >
    GREATER_EQ = 'GREATER_EQ' # >=
    EQUALS = 'EQUALS'       # =
    NOT_EQUALS = 'NOT_EQUALS' # ne
    
    # Punctuation
    LPAREN = 'LPAREN'       # (
    RPAREN = 'RPAREN'       # )
    SEMICOLON = 'SEMICOLON' # ;
    COMMA = 'COMMA'         # ,
    
    # Special operators
    CONDITIONAL = 'CONDITIONAL'  # ->
    BAR = 'BAR'             # |
    AMPERSAND = 'AMPERSAND' # &
    
    # Special tokens
    TAU = 'TAU'            # tau (tuple constructor)
    
    # End of file
    EOF = 'EOF'
    
    # Error
    ERROR = 'ERROR'

    # Additional token for period (.)
    PERIOD = 'PERIOD'       # .

class Token(NamedTuple):
    type: TokenType
    value: str
    line: int
    column: int

def rpal_lexer(input_string: str) -> List[Token]:
    """
    RPAL Lexical Analyzer
    
    Based on RPAL lexicon specification:
    - Identifier: Letter (Letter | Digit | '_')*
    - Integer: Digit+
    - String: ''' (Letter | Digit | Operator_symbol)* '''
    - Operators: +, -, *, /, **, <, <=, >, >=, =, <>, &, |, ->, etc.
    - Keywords: let, in, where, fn, rec, and, or, not, etc.
    """
    
    # Keywords mapping
    keywords = {
        'let': TokenType.LET,
        'in': TokenType.IN,
        'where': TokenType.WHERE,
        'fn': TokenType.FN,
        'rec': TokenType.REC,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'gr': TokenType.GR,
        'ge': TokenType.GE,
        'ls': TokenType.LS,
        'le': TokenType.LE,
        'eq': TokenType.EQ,
        'ne': TokenType.NE,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'nil': TokenType.NIL,
        'dummy': TokenType.DUMMY,
        'within': TokenType.WITHIN,
        'aug': TokenType.AUG,
        'conc': TokenType.IDENTIFIER,
        'stern': TokenType.STERN,
        'stem': TokenType.STEM,
        'Istuple': TokenType.ISTUPLE,
        'Isinteger': TokenType.ISINTEGER,
        'Isstring': TokenType.ISSTRING,
        'Isfunction': TokenType.ISFUNCTION,
        'Istruthvalue': TokenType.ISTRUTHVALUE,
        'Isdummy': TokenType.ISDUMMY,
        'Ito': TokenType.ITO,
        'Order': TokenType.ORDER,
        'Null': TokenType.NULL,
        'Print': TokenType.PRINT,
        'tau': TokenType.TAU,
    }
    
    # Token specification (order matters for longest match)
    token_specification = [
        # Comments (should be skipped)
        ('COMMENT', r'//.*'),
        
        # Multi-character operators (must come before single character ones)
        ('POWER', r'\*\*'),
        ('CONDITIONAL', r'->'),
        ('LESS_EQ', r'<='),
        ('GREATER_EQ', r'>='),
        ('NOT_EQUALS', r'<>'),
        
        # Single character operators and punctuation
        ('EQUALS', r'='),
        ('LESS', r'<'),
        ('GREATER', r'>'),
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('MULTIPLY', r'\*'),
        ('DIVIDE', r'/'),
        ('AT', r'@'),
        ('AMPERSAND', r'&'),
        ('BAR', r'\|'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('SEMICOLON', r';'),
        ('COMMA', r','),
        ('PERIOD', r'\.'),  # Added period token
        
        # Identifiers and Keywords (letters, digits, underscore)
        ('IDENTIFIER', r'[A-Za-z][A-Za-z0-9_]*'),
        
        # Integer literals
        ('INTEGER', r'\d+'),
        
        # String literals (enclosed in single quotes)
        ('STRING', r"'([^'\\]|\\.)*'"),
        
        # Whitespace (to be skipped)
        ('WHITESPACE', r'[ \t\n\r]+'),
        
        # Catch-all for unrecognized characters
        ('MISMATCH', r'.'),
    ]
    
    # Compile the regex
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    
    tokens = []
    line_num = 1
    line_start = 0
    
    for mo in re.finditer(tok_regex, input_string):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start + 1
        
        if kind == 'WHITESPACE':
            # Count newlines to track line numbers
            line_num += value.count('\n')
            if '\n' in value:
                line_start = mo.end() - len(value.split('\n')[-1])
            continue
        elif kind == 'COMMENT':
            # Skip comments
            continue
        elif kind == 'MISMATCH':
            tokens.append(Token(TokenType.ERROR, value, line_num, column))
        elif kind == 'IDENTIFIER':
            # Check if it's a keyword
            token_type = keywords.get(value.lower(), TokenType.IDENTIFIER)
            tokens.append(Token(token_type, value, line_num, column))
        else:
            # Map the token kind to TokenType
            token_type_map = {
                'POWER': TokenType.POWER,
                'CONDITIONAL': TokenType.CONDITIONAL,
                'LESS_EQ': TokenType.LESS_EQ,
                'GREATER_EQ': TokenType.GREATER_EQ,
                'NOT_EQUALS': TokenType.NOT_EQUALS,
                'EQUALS': TokenType.EQUALS,
                'LESS': TokenType.LESS,
                'GREATER': TokenType.GREATER,
                'PLUS': TokenType.PLUS,
                'MINUS': TokenType.MINUS,
                'MULTIPLY': TokenType.MULTIPLY,
                'DIVIDE': TokenType.DIVIDE,
                'AT': TokenType.AT,
                'AMPERSAND': TokenType.AMPERSAND,
                'BAR': TokenType.BAR,
                'LPAREN': TokenType.LPAREN,
                'RPAREN': TokenType.RPAREN,
                'SEMICOLON': TokenType.SEMICOLON,
                'COMMA': TokenType.COMMA,
                'PERIOD': TokenType.PERIOD,  # Added period token
                'INTEGER': TokenType.INTEGER,
                'STRING': TokenType.STRING,
            }
            token_type = token_type_map.get(kind, TokenType.ERROR)
            tokens.append(Token(token_type, value, line_num, column))
    
    # Add EOF token
    tokens.append(Token(TokenType.EOF, 'EOF', line_num, len(input_string) - line_start + 1))
    
    return tokens

# ------------------------ Testing ------------------------

if __name__ == '__main__':
    # Test RPAL programs
    test_programs = [
        "let rec Rev S = S eq '' -> '' | (Rev(Stern S)) @Conc (Stem S ) in let Pairs (S1,S2) = P (Rev S1, Rev S2) where rec P (S1, S2) = S1 eq '' & S2 eq '' -> nil |  (fn L. P (Stern S1, Stern S2) aug ((Stem S1) @Conc (Stem S2))) nil in Print ( Pairs ('abc','def'))"
    ]
    
    for i, program in enumerate(test_programs, 1):
        print(f"\n🔍 Test Program {i}: {program}")
        print("=" * 50)
        
        try:
            tokens = rpal_lexer(program)
            
            for token in tokens:
                if token.type == TokenType.EOF:
                    print(f"({token.type.value}, {token.value})")
                else:
                    print(f"({token.type.value}, '{token.value}') at line {token.line}, col {token.column}")
                    
        except Exception as e:
            print(f"Error: {e}")
        
        print()

    
