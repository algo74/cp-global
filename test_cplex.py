
from sys import stdout
import numpy as np
from docplex.cp.model import CpoModel
import matplotlib.pyplot as plt
import requests
from io import BytesIO

import cplex_context


def display(sol):
    
    chess_board = np.zeros((NB_QUEEN, NB_QUEEN, 3))
    black = 0.5
    white = 1
    for l in range(NB_QUEEN):
        for c in range(NB_QUEEN):
            if (l%2 == c%2):
                col = white
            else:
                col = black
            chess_board[l,c,::]=col

    fig, ax = plt.subplots(figsize=(NB_QUEEN / 2, NB_QUEEN / 2))
    ax.imshow(chess_board, interpolation='none')
    # wq_im_file = "./n_queen_utils/WQueen.png"
    # bq_im_file = "./n_queen_utils/BQueen.png"
    # wq_im_file = "https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/cp/jupyter/n_queen_utils/WQueen.png?raw=true"
    # bq_im_file = "https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/cp/jupyter/n_queen_utils/BQueen.png?raw=true"
    # wq = plt.imread(wq_im_file)
    # bq = plt.imread(bq_im_file)
    wq_im_url = "https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/cp/jupyter/n_queen_utils/WQueen.png?raw=true"
    bq_im_url = "https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/cp/jupyter/n_queen_utils/BQueen.png?raw=true"

    response_wq = requests.get(wq_im_url)
    response_bq = requests.get(bq_im_url)

    wq = plt.imread(BytesIO(response_wq.content))
    bq = plt.imread(BytesIO(response_bq.content))
    for y, x in enumerate(sol):
        if (x%2 == y%2):
            queen = bq
        else:
            queen = wq 
        ax.imshow(queen, extent=[x-0.4, x + 0.4, y - 0.4, y + 0.4])
    ax.set(xticks=[], yticks=[])
    ax.axis('image')
    plt.show()

if __name__ == "__main__":
    print("Hello, world!")
    NB_QUEEN = 8
    mdl = CpoModel(name='NQueens')
    x = mdl.integer_var_list(NB_QUEEN, 0, NB_QUEEN - 1, "X")
    mdl.add(mdl.all_diff(x))
    mdl.add(mdl.all_diff(x[i] + i for i in range(NB_QUEEN)))
    mdl.add(mdl.all_diff(x[i] - i for i in range(NB_QUEEN)))
    msol = mdl.solve(TimeLimit=10, agent=cplex_context.CPLEX_AGENT, execfile=cplex_context.CPLEX_EXECUTABLE_PATH)
    if msol: 
      stdout.write("Solution:")
      sol = [msol[v] for v in x]
      for v in range(NB_QUEEN):
        stdout.write(" " + str(sol[v]))
      stdout.write("\n")
      stdout.write("Solve time: " + str(msol.get_solve_time()) + "\n")
      display(sol)
    else:
      stdout.write("No solution found\n")

