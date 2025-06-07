"""
CSE Factory Module

This module provides the main interface for evaluating RPAL programs.
It coordinates the parsing, standardization, and interpretation phases
of RPAL program execution.

Key components:
- AST generation from source code
- Standardization of AST to ST (Standard Tree)
- CSE machine adaptation and interpretation
- Output management
"""

from parser import parse_rpal
from Standardizer import ast_to_st
from rpal_interpreter import RPALInterpreter
from st_cse_adapter import adapt_st_for_cse

def evaluate_rpal(source, ast_show=False, st_show=False):
    """
    Main function to evaluate RPAL source code.
    
    Process flow:
    1. Parses source into Abstract Syntax Tree (AST)
    2. Converts AST to Standardized Tree (ST)
    3. Adapts ST for CSE machine execution
    4. Interprets the adapted ST using RPAL interpreter
    
    Args:
        source: String containing RPAL source code
        ast_show: Boolean flag to print AST (default: False)
        st_show: Boolean flag to print ST (default: False)
        
    Returns:
        None - prints program output to console
        
    Example:
        evaluate_rpal("let X = 5 in X", ast_show=True)
    """
    # Generate Abstract Syntax Tree
    ast = parse_rpal(source)
    if ast_show:
        ast.print_tree(0)
    
    # Convert to Standardized Tree
    st = ast_to_st(ast)
    if st_show:
        st.print_tree(0)
    
    # Adapt ST for CSE machine
    adapted_st = adapt_st_for_cse(st)
    
    # Create interpreter and execute program
    interpreter = RPALInterpreter()
    interpreter.interpret(adapted_st)
    
    # Get and display output
    output = interpreter.get_output().strip()
    print(output)

def __init__():
    """
    Module initialization function.
    Currently empty as no initialization is needed.
    """
    pass

