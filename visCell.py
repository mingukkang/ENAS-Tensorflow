import pygraphviz as pgv
import numpy as np
import sys

def construct_block(graph, num_block, ops):

    ops_name = ["conv 3x3", "conv 5x5", "avg pool", "max pool", "identity", "add", "concat"]

    for i in range(0, 2):
        graph.add_node(num_block*10+i+1,
                       label="{}".format(ops_name[ops[2*i+1]]),
                       color='black',
                       fillcolor='yellow',
                       shape='box',
                       style='filled')

    #graph.add_subgraph([num_block*10+1, num_block*10+2], rank='same')

    graph.add_node(num_block*10+3,
                   label="Add",
                   color='black',
                   fillcolor='greenyellow',
                   shape='box',
                   style='filled')

    graph.add_subgraph([num_block*10+1, num_block*10+2, num_block*10+3],
                       name='cluster_s{}'.format(num_block))

    for i in range(0, 2):
        graph.add_edge(num_block*10+i+1, num_block*10+3)

def connect_block(graph, num_block, ops, output_used):

    for i in range(0, 2):
        graph.add_edge(ops[2*i]*10+3, (num_block*10)+i+1)
        output_used.append(ops[2*i]*10+3)

def creat_graph(cell_arc):

    G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open', rankdir='TD')

    #creat input
    G.add_node(3, label="H[i-1]", color='black', shape='box')
    G.add_node(13, label="H[i]", color='black', shape='box')
    G.add_subgraph([3, 13], name='cluster_inputs', rank='same', rankdir='TD', color='white')

    #creat blocks
    for i in range(0, len(cell_arc)):
        construct_block(G, i+2, cell_arc[i])

    #connect blocks to each other
    output_used = []
    for i in range(0, len(cell_arc)):
        connect_block(G, i+2, cell_arc[i], output_used)

    #creat output
    G.add_node((len(cell_arc)+2)*10+3,
               label="Concat",
               color='black',
               fillcolor='pink',
               shape='box',
               style='filled')

    for i in range(0, len(cell_arc)+2):
        if not(i*10+3 in output_used) :
            G.add_edge(i*10+3, (len(cell_arc)+2)*10+3)

    return G


def main():

    if(len(sys.argv) <= 1):
        norm_cell = "0 1 0 1 2 0 1 1 3 4 3 1 4 4 1 1 5 1 1 1"
        redu_cell = "0 0 0 1 2 3 1 0 2 0 2 2 3 0 1 4 2 2 5 4"
    else:
        norm_cell, redu_cell = "", ""
        for i in range(1, len(sys.argv)/2+1):
            norm_cell += "{} ".format(sys.argv[i])
        for i in range(len(sys.argv)/2+1, len(sys.argv)):
            redu_cell += "{} ".format(sys.argv[i])
        print("{}\n{}".format(norm_cell, redu_cell))

    ncell = np.array([int(x) for x in norm_cell.split(" ") if x])
    rcell = np.array([int(x) for x in redu_cell.split(" ") if x])

    ncell = np.reshape(ncell, [-1, 4])
    rcell = np.reshape(rcell, [-1, 4])

    Gn = creat_graph(ncell)
    Gr = creat_graph(rcell)

    Gn.write("ncell.dot")
    Gr.write("rcell.dot")

    vizGn = pgv.AGraph("ncell.dot")
    vizGr = pgv.AGraph("rcell.dot")

    vizGn.layout(prog='dot')
    vizGr.layout(prog='dot')

    vizGn.draw("ncell.png")
    vizGr.draw("rcell.png")


if __name__ == '__main__':
    main()
