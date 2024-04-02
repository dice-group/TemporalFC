from utils_TP.static_funcs import calculate_wilcoxen_score, select_model
from main import argparse_default

args = argparse_default()
model, frm = select_model(args)
print("test")
