"""
RPAL Language Interpreter
This is the main entry point for the RPAL interpreter. It handles the complete process of
lexical analysis, parsing, standardization, and execution of RPAL programs.
"""

import argparse
from src.lexer import Lexer
from src.parser import Parser
from src.standerizer.ast import AST
from src.standerizer.node import Node
from src.standerizer import ast_factory
from src.lcrs_to_nary_convertor import lcrs_to_nary
from src.rpal_ast import print_ast
from src.nary_to_lcrs_convertor import nary_to_lcrs
from src.cse_machine.machine import CSEMachine

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RPAL Language Interpreter")
    parser.add_argument("filename", help="Input RPAL program file")
    parser.add_argument("-ast", action="store_true", help="Print original AST only")
    parser.add_argument("-st", action="store_true", help="Print standardized AST only")
    args = parser.parse_args()

    # Read source code from file
    with open(args.filename, "r") as file:
        source_code = file.read()

    # Lexical Analysis: Convert source code to tokens
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()

    # Syntax Analysis: Build Abstract Syntax Tree
    parser = Parser(tokens)
    ast_root = parser.parse()

    # Option 1: Print original AST and exit
    if args.ast:
        print_ast(ast_root)
        return

    # Convert LCRS AST to n-ary tree for standardization
    nary_root = lcrs_to_nary(ast_root)
    ast_obj = AST(nary_root)

    # Standardize the AST according to RPAL rules
    ast_obj.standardize()

    # Option 2: Print standardized AST and exit
    if args.st:
        ast_obj.print_ast()
        return


    
    # Execute the program using CSE machine
    cse_machine = CSEMachine()
    cse_machine.execute(ast_obj.root)
    
    # Print the program output
    print(cse_machine._generate_output())


if __name__ == "__main__":
    main()
