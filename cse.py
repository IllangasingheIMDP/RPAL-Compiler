tests=["let rec Rev S = S eq '' -> '' | (Rev(Stern S)) @Conc (Stem S ) within Pairs (S1,S2) =  not (Isstring S1 & Isstring S2) -> 'both args not strings' | P (Rev S1, Rev S2) where rec P (S1, S2) =  S1 eq '' & S2 eq '' -> nil | (Stern S1 eq '' & Stern S2 ne '') or (Stern S1 ne '' & Stern S2 eq '') -> 'bad strings' | (P (Stern S1, Stern S2) aug ((Stem S1) @Conc (Stem S2))) in Print ( Pairs ('abc','def'))"]
from parser import parse_rpal
from Standardizer import ast_to_st
from cse_machine.machine import CSEMachine
from st_cse_adapter import adapt_st_for_cse

ast = parse_rpal(tests[0])
st = ast_to_st(ast)
adapted_st = adapt_st_for_cse(st)
cse_machine = CSEMachine()
cse_machine.execute(adapted_st)
print(cse_machine._generate_output())

