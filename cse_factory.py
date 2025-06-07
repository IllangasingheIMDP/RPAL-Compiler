from parser import parse_rpal
from Standardizer import ast_to_st
from machine import CSEMachine
from st_cse_adapter import adapt_st_for_cse



def evaluate_rpal(source, ast_show=False,st_show=False):
    
    ast = parse_rpal(source)
    if ast_show:
        ast.print_tree(0)
    st = ast_to_st(ast)
    if st_show:
        st.print_tree(0)
    adapted_st = adapt_st_for_cse(st)
    cse_machine = CSEMachine()
    cse_machine.execute(adapted_st)
    print(cse_machine._generate_output())
def __init__():
    pass
    
