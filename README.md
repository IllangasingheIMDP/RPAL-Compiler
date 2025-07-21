# RPAL-Compiler

A Python-based compiler and interpreter for the RPAL (Right-reference Programming Algorithmic Language), supporting the full translation pipeline from parsing source code to program execution.

## Features

### 1. Lexical Analysis
- **Custom Lexer:** The project includes a hand-written lexer that scans RPAL source code and produces a stream of tokens, handling identifiers, literals, keywords, operators, and special symbols according to the RPAL language specification.

### 2. Parsing
- **Recursive Descent Parser:** The parser processes the token stream to build an Abstract Syntax Tree (AST) that represents the syntactic structure of the input program. It supports the full RPAL grammar, including let-expressions, functions, conditionals, recursion, tuples, and more.
- **AST Visualization:** You can print the generated AST for any input program using the `-ast` option, useful for debugging and understanding code structure.

### 3. Standardization
- **AST to Standard Tree (ST):** The `Standardizer` module transforms the AST into a Standardized Tree. This process simplifies and normalizes the syntax, making the program easier to interpret and reducing the number of constructs the interpreter must handle.
- **ST Visualization:** You can print the standardized tree using the `-st` option for insight into the standardization process.

### 4. Control Structure Execution (CSE) Machine
- **Linearization:** Converts the standardized tree into a sequence of control structures suitable for stack-based execution.
- **Interpreter Engine:** The CSE engine executes these control structures, handling:
    - Variable binding and lexical scoping
    - Function application (including higher-order and recursive functions)
    - Built-in operations (arithmetic, logic, string processing, tuples)
    - Conditional expressions and pattern matching
    - Tuple construction and deconstruction
    - Print and type-checking functions

### 5. Built-in Functions & Operations
- **Arithmetic:** Addition, subtraction, multiplication, division, exponentiation, negation.
- **Logic:** And, or, not, equality, inequality, comparison (>, >=, <, <=).
- **Tuples:** Construction (`aug`), concatenation, order (size), and access.
- **Strings:** Concatenation, conversion, and character access (`Stem`/`Stern`).
- **Type Checks:** `Isstring`, `Isinteger`, `Istruthvalue`, `Isfunction`, `Null`, `Istuple`.
- **Printing:** Outputs program results using the `Print` operation.

### 6. Error Handling
- **Detailed Error Messages:** The interpreter and parser generate informative error messages for syntax errors, semantic errors, and runtime exceptions, aiding debugging and learning.

### 7. Utilities & Tooling
- **Makefile Support:** Common workflows (run, print AST, clean) are streamlined via the included Makefile.
- **Test Programs:** Example RPAL programs are included for demonstration and testing.

## Usage

The main entry point is `myrpal.py`. You can run and interact with RPAL files using the provided Makefile or directly with Python.

### Running with Python

```bash
python myrpal.py [options] filename
```

**Options:**
- `-ast` : Show the Abstract Syntax Tree for the RPAL file.
- `-st`  : Show the Standardized Tree for the RPAL file.

**Example:**
```bash
python myrpal.py -ast test.rpal
python myrpal.py -st test.rpal
python myrpal.py test.rpal
```

### Using the Makefile

Several utility commands are provided:

- `make run filename` : Run the interpreter with an RPAL file.
- `make ast filename` : Show only the AST for an RPAL file.
- `make clean`        : Clean up compiled files.
- `make help`         : Show command help.

**Example:**
```bash
make run test.rpal
make ast test.rpal
make clean
```

## Project Structure

- `myrpal.py`        : Main script to parse and interpret RPAL programs.
- `parser.py`        : Implements the RPAL lexer and parser, producing an AST.
- `Standardizer.py`  : Converts ASTs into Standardized Trees.
- `cse_factory.py`   : Coordinates parsing, standardization, and execution.
- `rpal_interpreter.py` : Implements the CSE interpreter for executing programs.

## About RPAL

RPAL is a functional programming language designed for educational purposes, emphasizing the transformation from source code to executable semantics through parsing, standardization, and interpretation.

## License

This project currently does not specify a license.
