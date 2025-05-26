#!/usr/bin/env python3
import sys
from lexical import rpal_lexer
from parser import parse_rpal
from StAndCSE import evaluate_rpal

def main():
    # Check if -ast switch is provided
    ast_only = False
    file_name = None
    
    for arg in sys.argv[1:]:
        if arg == "-ast":
            ast_only = True
        else:
            file_name = arg
    
    if not file_name:
        print("Usage: python myrpal.py [-ast] file_name")
        sys.exit(1)
    
    try:
        with open(file_name, 'r') as file:
            source = file.read()
            evaluate_rpal(source, ast_only)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
