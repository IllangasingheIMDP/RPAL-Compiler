from parser import parse_rpal
from Standardizer import ast_to_st
from rpal_interpreter import RPALInterpreter
from st_cse_adapter import adapt_st_for_cse



def evaluate_rpal(source, ast_show=False,st_show=False):
    
    ast = parse_rpal(source)
    if ast_show:
        ast.print_tree(0)
    st = ast_to_st(ast)
    if st_show:
        st.print_tree(0)
    adapted_st = adapt_st_for_cse(st)
    interpreter = RPALInterpreter()
    interpreter.interpret(adapted_st)
    output = interpreter.get_output().strip()
    print(output)
def __init__():
    pass
    
