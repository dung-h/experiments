import networkx as nx

# Line
def line(n):
    g = nx.Graph()
    g.add_nodes_from(list(range(n)))
    for i in range(n-1):
        g.add_edge(i, i+1)
    return g

# Ring
def ring(n):
    g = nx.Graph()
    g.add_nodes_from(list(range(n)))
    for i in range(n):
        g.add_edge(i, (i+1) % n)
    return g

# Star
def star(n):
    g = nx.Graph()
    g.add_nodes_from(list(range(n)))
    for i in range(1, n):
        g.add_edge(0, i)
    return g

# Q-grid
def qgrid(m,n):
    g = nx.Graph()
    g.add_nodes_from(list(range(0,m*n-1)))
    for i in range(0,m):
        for j in range(0,n):
            if i < m-1: g.add_edge(j*m+i, j*m+i+1)
            if j < n-1: g.add_edge(j*m+i, (j+1)*m+i)
    return g

# IBM Q Ourense (5 qubits)
def ourense():
    g = nx.Graph()
    g.add_nodes_from(list(range(5)))
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(0,3)
    g.add_edge(3,4)
    return g

# IBM Q Tokyo (20 qubits) 
def tokyo():
    g = nx.Graph()
    g.add_nodes_from(list(range(20)))
    for i in range(0,4):
        g.add_edge(i,i+1)
        g.add_edge(i+1,i)

    for i in range(5,9):
        g.add_edge(i,i+1)
        g.add_edge(i+1,i)
        
    for i in range(10,14):
        g.add_edge(i,i+1)
        g.add_edge(i+1,i)
        
    for i in range(15,19):
        g.add_edge(i,i+1)
        g.add_edge(i+1,i)
        
    for i in range(0,15):
        g.add_edge(i,i+5)
        g.add_edge(i+5,i)

    for i in [1,3,5,7,11,13]:
        g.add_edge(i,i+6)
        g.add_edge(i+6,i)

    for i in [2,4,6,8,12,14]:
        g.add_edge(i,i+4)
        g.add_edge(i+4,i)
    return g

# IBM Q Rochester (53 qubits)
def rochester():
    g = nx.Graph()
    g.add_nodes_from(list(range(53)))
    I_1= list(range(4)) + list(range(7,15)) +\
        list(range(19,27)) + list(range(30,38)) +\
            list(range(42,50))
            
    for i in I_1:
        g.add_edge(i,i+1)
    E = [(0,5),(5,9),(4,6),(6,13),(7,16),(16,19),\
         (11,17),(17,23),(15,18),(18,27),(21,28),(28,32),\
             (25,29),(29,36),(30,39),(39,42),(34,40),(40,46),\
                 (38,41),(41,50),(44,51),(48,52)]
    g.add_edges_from(E)
    return g

# Google Sycamore (53 qubits)
def sycamore53():
    g = nx.Graph()
    g.add_nodes_from(list(range(54))) 
    I = list(range(6,12))+list(range(18,24))+list(range(30,36))+\
        list(range(42,48))
    for i in I:
        for j in g.nodes():
            if j in I: continue
            if i-j in [5,6] or j-i in [6,7]:
                g.add_edge(i,j)
    g.remove_node(3)
    assert 3 not in g.nodes(), 'Node error'
    mapping = dict()
    for n in g.nodes():
        if n < 3:
            mapping[n] = n
        else:
            mapping[n] = n - 1
            
    h = nx.relabel_nodes(g, mapping)
    return h

# Google Sycamore (54 qubits)
def sycamore54():
    g = nx.Graph()
    g.add_nodes_from(list(range(54)))
    I = list(range(6, 12))+list(range(18, 24))+list(range(30, 36))+list(range(42, 48))
    for i in I:
        for j in g.nodes():
            if j in I:
                continue
            if i-j in [5, 6] or j-i in [6, 7]:
                g.add_edge(i, j)
    return g

def eagle():
    g = nx.Graph()
    g.add_nodes_from(list(range(127))) 
    eagle_cg = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (12,13),
        (0,14), (14,18), (4,15), (15,22), (8,16), (16,26), (12,17), (17,30),
        (18,19), (19,20), (20,21), (21,22), (22,23), (23,24), (24,25), (25,26), (26,27), (27,28), (28,29), (29,30), (30,31), (31,32),
        (20,33), (33,39), (24,34), (34,43), (28,35), (35,47), (32,36), (36,51),
        (37,38), (38,39), (39,40), (40,41), (41,42), (42,43), (43,44), (44,45), (45,46), (46,47), (47,48), (48,49), (49,50), (50,51),
        (37,52), (52,56), (41,53), (53,60), (45,54), (54,64), (49,55), (55,68),
        (56,57), (57,58), (58,59), (59,60), (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,68), (68,69), (69,70),
        (58,71), (71,77), (62,72), (72,81), (66,73), (73,85), (70,74), (74,89),
        (75,76), (76,77), (77,78), (78,79), (79,80), (80,81), (81,82), (82,83), (83,84), (84,85), (85,86), (86,87), (87,88), (88,89),
        (75,90), (90,94), (79,91), (91,98), (83,92), (92,102), (87,93), (93,106),
        (94,95), (95,96), (96,97), (97,98), (98,99), (99,100), (100,101), (101,102), (102,103), (103,104), (104,105), (105,106), (106,107), (107,108),
        (96,109), (100,110), (110,118), (104,111), (111,122), (108,112), (112,126),
        (113,114), (114,115), (115,116), (116,117), (117,118), (118,119), (119,120), (120,121), (121,122), (122,123), (123,124), (124,125), (125,126),
    ]  # ibm_washington coupling map from quantum-computing.ibm.com
    g.add_edges_from(eagle_cg)
    return g

def heron():
    g = nx.Graph()
    g.add_nodes_from(list(range(156))) 

    heron_cg = []

    # Add edges in horizontal rows
    for row in range(8):
        for col in range(15): # Stop short because right-most node has no edges further right
            heron_cg.append(((row*20)+col, (row*20)+col+1))

    # Add vertical connections between rows
    for row in range(7): # Stop one short because bottom row has no edges below it
        if (row % 2) == 0:
            # Num represents number of vertical links, going left to right
            for num, col in zip(range(4), range(3, 16, 4)):
                heron_cg += [ ((row*20)+16+num, (row*20)+col), ((row*20)+16+num, ((row+1)*20)+col)]
        else:
            for num, col in zip(range(4), range(1, 14, 4)):
                heron_cg += [ ((row*20)+16+num, (row*20)+col), ((row*20)+16+num, ((row+1)*20)+col)]

    g.add_edges_from(heron_cg)
    return g


if __name__=='__main__':
    AG = sycamore54()
    print(AG.edges())