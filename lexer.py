# test_boolean_parser.py

import re
from parser import BooleanParser

# ------------------------ Lexer ------------------------
def lexer(input_string):
    token_specification = [
        ('OR', r'\bor\b'),
        ('NOR', r'\bnor\b'),
        ('XOR', r'\bxor\b'),
        ('AND', r'\band\b'),
        ('NAND', r'\bnand\b'),
        ('NOT', r'\bnot\b'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('TRUE', r'\btrue\b'),
        ('FALSE', r'\bfalse\b'),
        ('ID', r'\bi\b'),
        ('SKIP', r'[ \t]+'),
        ('MISMATCH', r'.')
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, input_string):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Unexpected character: {value!r}')
        else:
            tokens.append((kind, value))
    tokens.append(('EOF', 'EOF'))
    return tokens

# ------------------------ Testing ------------------------

if __name__ == '__main__':
    # Test input string
    input_string = 'true nand (false xor i) or not i and not false nor i'

    # Step 1: Lexical Analysis
    token_list = lexer(input_string)
    print("🔍 Tokens:")
    for token in token_list:
        print(token)