import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import random
import os
import argparse
from pathlib import Path
import re
from scipy.io import mmread
import matplotlib.lines as mlines
import json
from braket.aws import AwsDevice
from braket.ahs.atom_arrangement import AtomArrangement
from quera_ahs_utils.plotting import show_register, show_global_drive, show_final_avg_density
from quera_ahs_utils.drive import get_drive 
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.devices import LocalSimulator
import time
from braket.aws import AwsSession, AwsQuantumTask
from utils import graph_io

def maximum_weighted_independent_set_qubo(G, weight, penalty_coeff):
    if not G:
        return {}
    cost = dict(G.nodes(data=weight, default=1))
    scale = max(cost.values())
    Q = {(node, node): min(-cost[node] / scale, 0.0) for node in G}
    Q.update({edge: penalty_coeff for edge in G.edges})

    #Need to switch from QUBO to Ising: x_i = (1-s_i)/2
    for i in range(len(G.nodes)):
        Q[(i,i)] *= -0.5
    for key in Q:
        if key[0] != key[1]:
            Q[key] *= 0.25
            Q[(key[0], key[0])] -= penalty_coeff * 0.25
            Q[(key[1], key[1])] -= penalty_coeff * 0.25

    Q = {x:y for x,y in Q.items() if y!=0}

    return Q

def maximum_weighted_independent_set_qubo_openqaoa(G, weight=None, penalty_coeff = 2.0):
    Q = maximum_weighted_independent_set_qubo(G,weight,penalty_coeff)

    terms = []
    weights = []
    for term in Q:
        if term[0] == term[1]:
            terms.append((term[0],))
            weights.append(Q[term])
        else:
            terms.append((term[0],term[1]))
            weights.append(Q[term])
    mis_qubo = QUBO(len(G.nodes), terms, weights)

    return mis_qubo

def parse_qaoa_result_string(s):
    ind_set = []
    for i in range(len(s)):
        if s[i] == '1':
            ind_set.append(i)
    return ind_set

def parse_dwave_result(sample):
    sol = []
    for i in sample:
        if sample[i] == -1:
            sol.append(int(i))
    return sol

def repair_dwave_ind_sol(G, ind_set):
    if is_independent_set(G, ind_set):
        return ind_set
    else:
        while is_independent_set(G, ind_set) == False:
            node_with_largest_degree = -1
            largest_degree = -1
            for node in ind_set:
                if G.degree[node] > largest_degree:
                    node_with_largest_degree = node
                    largest_degree = G.degree[node]
            ind_set.remove(node_with_largest_degree)
        return ind_set
    
def check_quera_results(results):
    wrong_num = 0
    print(len(results["measurements"]))
    for measurement in results["measurements"]: # For each measurement...
        pre_sequence = np.array(measurement["pre_sequence"])
        if np.any(pre_sequence==0): 
            wrong_num += 1 # skip anyshots with defects
    if wrong_num == len(results["measurements"]):
        return False
    else:
        return True
    
def repair_quera_ind_sol(G,results):      
    ind_sets = []
    for measurement in results["measurements"]: # For each measurement...
        pre_sequence = np.array(measurement["pre_sequence"])
        post_sequence = np.array(measurement["post_sequence"])
        if np.any(pre_sequence==0): continue # skip anyshots with defects
        

        bitstring = post_sequence

        if bitstring.size == 0:
            continue
        ind_set = list(np.nonzero(bitstring==0)[0])    # Find indices of IS vertices
        if is_independent_set(G, ind_set):
            ind_sets.append(ind_set)
        else:
            while is_independent_set(G, ind_set) == False:
                node_with_largest_degree = -1
                largest_degree = -1
                for node in ind_set:
                    if G.degree[node] > largest_degree:
                        node_with_largest_degree = node
                        largest_degree = G.degree[node]
                ind_set.remove(node_with_largest_degree)
            ind_sets.append(ind_set)

        
        
    if len(ind_sets) == 0: 
        raise ValueError("no independent sets found! increase number of shots.")
        
    return ind_sets

def kings_graph(numx,numy,filling=0.7,seed=None):
    '''
    Generate a next nearest neighbor graph with a lattice constant 1, with some number of nodes removed
    numx    - number of grid points in the X direction
    numy    - number of grid points in the Y direction
    filling - Fraction of vertices to be kept. Total number of vertices is int(numx*numy*filling)
    
    Returns
    pos     - [N x 2] array of points on a square grid
    graph   - networkx connectivity graph
    '''
    xx,yy = np.meshgrid(range(numx),range(numy))
    num_points = int(numx*numy*filling)
    rand = np.random.default_rng(seed=seed)
    # Generate points
    points = np.array([xx.flatten(),yy.flatten()]).T
    points = points[rand.permutation(numx*numy)[0:num_points],:]
    # Generate a unit disk graph by thresholding distances between points.
    distances = np.sqrt((points[:,0] - points[:,0,None])**2 + (points[:,1] - points[:,1,None])**2)
    graph     = nx.Graph(distances<np.sqrt(2)+1E-10)
    
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return points, graph

def visualize_graph(ax,graph,positions,node_colors = "#6437FF"):
    '''
    Visualize a graph using networkx
    ax          - matplotlib axis to draw on
    graph       - networkx graph of vertices and edges, with vertex labels as integers
    positions   - Positions of each vertex
    node_colors - Color of nodes. Either a string, or list of strings, one for each vertex
    '''
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    pos_dict = {a:positions[a] for a in range(positions.shape[0])}
    nx.draw_networkx_edges(graph,pos_dict,width=10/np.sqrt(len(graph.nodes)))
    nx.draw_networkx_nodes(graph,pos_dict,node_size=600/np.sqrt(len(graph.nodes)),node_color=node_colors)

def find_UDG_radius(position, graph):
    '''
    Computes the optimal unit disk radius for a particular set of positions and graph.
    position   - [N x 2] array of points
    graph       - network connectivity graph. This should be a unit disk graph.
    
    returns
    radius      - Optimal unit disk radius of the graph
    rmin        - Minimum distance
    rmax        - Maximum distance
    '''
    
    dists = np.sqrt((position[:,0,None] - position[:,0])**2
               + (position[:,1,None] - position[:,1])**2)
    rmin = 0
    rmax = np.inf
    for i in range(position.shape[0]):
        for j in range(i+1,position.shape[0]):
            if (i,j) in graph.edges:
                if rmin<dists[i,j]:
                    rmin = dists[i,j]
            elif (i,j) not in graph.edges:
                if rmax>dists[i,j]:
                    rmax = dists[i,j]
    
    if rmin>rmax:
        position_dict = {}
        for i in range(len(position)):
            position_dict[i] = position[i]
        nx.draw(graph, pos=position_dict, node_size=20)
        plt.savefig('graph.png')
        exit()
        raise BaseException("Graph is not a unit disk graph!")
    
    if rmax == np.inf:
        rmax = rmin
    
    return np.sqrt(rmin*rmax),rmin,rmax

def save_result_json(json_file,result):
    '''
    Helper function to save results locally
    '''
    result_dict = {"measurements":[]}
    for measurement in result.measurements:
        shot_result = {
            "pre_sequence":[int(qubit) for qubit in measurement.pre_sequence],
            "post_sequence":[int(qubit) for qubit in measurement.post_sequence]
                      } 
        result_dict["measurements"].append(shot_result)
        
    with open(json_file,"w") as io:
        json.dump(result_dict,io,indent=2)
        
def open_json(json_file):
    '''
    Helper function to load and open json data
    '''
    with open(json_file,"r") as io:
        return json.load(io) 

def postprocess_MIS(G,results):
    '''
    Removes vertices that violate the independent set condition
    G - networkx graph
    results - an AWS AnalogHamiltonianSimulationQuantumTaskResult
    
    returns
    data_out - a list of bitstrings which are valid independent sets of G
    '''
    data_out = []
    for measurement in results["measurements"]: # For each measurement...
        pre_sequence = np.array(measurement["pre_sequence"])
        post_sequence = np.array(measurement["post_sequence"])
        if np.any(pre_sequence==0): continue # skip anyshots with defects
        
        
        bitstring = post_sequence

        if bitstring.size == 0:
            continue
        inds = np.nonzero(bitstring==0)[0]    # Find indices of IS vertices
        subgraph = nx.subgraph(G,inds)        # Generate a subgraph from those vertices. If the bitstring is an independent set, this subgraph has no edges.
        inds2 = nx.maximal_independent_set(subgraph,seed=0) # Find the mIS of this subgraph. If there are no edges, it is the original bitstring. Else, it randomly chooses within each graph.
        payload = np.ones(len(bitstring))     # Forge into the correct data structure (a list of 1s and 0s)
        payload[inds2] = 0
        data_out.append(payload)
        
    if len(data_out) == 0: 
        raise ValueError("All inits are wrong! increase number of shots.")
        
    return np.asarray(data_out)

def analysis_MIS(graph,result_json):
    '''
    Helper function to analyze a MIS result and plot data
    '''

    post_bitstrings = np.array([q["post_sequence"] for q in result_json["measurements"]])
    pp_bitstrings = postprocess_MIS(graph, result_json)


    IS_sizes = np.sum(1-pp_bitstrings,axis=1)
    unique_IS_sizes,counts = np.unique(IS_sizes,return_counts=True)


    # avg_no_pp = 'Average pre-processed size:  {:0.4f}'.format( (1-post_bitstrings).sum(axis=1).mean() )
    # avg_pp = 'Average post-processed IS size: {:0.4f}'.format(IS_sizes.mean())
    # print(avg_no_pp)
    # print(avg_pp)
    # plt.bar(unique_IS_sizes,counts/counts.sum())
    # plt.xticks(unique_IS_sizes)
    # plt.xlabel("IS sizes",fontsize=14)
    # plt.ylabel("probability",fontsize=14)
    # plt.show()
    
    return IS_sizes,pp_bitstrings


def solve_local_mis(G, use_optimal, pos=None):
    if len(G.nodes) == 1:
        ind_set = G.nodes()
    elif len(G.nodes) == 0:
        ind_set = []
    else:
        if use_optimal == 'dwave':
            Q = maximum_weighted_independent_set_qubo(G, weight=None, penalty_coeff = 2.0)
            h = {}
            J = {}
            for key in Q:
                if key[0] == key[1]:
                    h[key[0]] = Q[key]
                else:
                    J[(key[0], key[1])] = Q[key]
            sampler = EmbeddingComposite(DWaveSampler())
            # sampleset = sampler.sample_ising(h, J, num_reads=1000)
            # print(sampleset.first)
            # ind_set = parse_dwave_result(sampleset.first.sample)
            ind_set = maximum_weighted_independent_set(G, sampler = sampler, lagrange=2.0, num_reads=1000)
            ind_set = repair_dwave_ind_sol(G, ind_set)
        elif use_optimal == 'exact':
            mis_qubo = maximum_weighted_independent_set_qubo_openqaoa(G)
            mis_hamil = mis_qubo.hamiltonian
            energy, configuration = ground_state_hamiltonian(mis_hamil)
            ind_set = parse_qaoa_result_string(configuration[0])
        elif use_optimal == 'kamis':
            ind_size, ind_set = run_kamis(G)
        elif use_optimal == 'ahs_local':
            qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
            capabilities = qpu.properties.paradigm
            C6 = float(capabilities.rydberg.dict()['c6Coefficient'])
            unitdisk_radius,min_radius,max_radius = find_UDG_radius(pos,G)
            Delta_final = 20e6 # rad/sec
            blockade_radius = (C6/(Delta_final))**(1/6)
            a = blockade_radius / unitdisk_radius
            if a < 4e-6:
                print("Forcing a to be 4e-6, hopefully it works...")
                a = 4e-6
            small_register = AtomArrangement()
            for x in pos:
                small_register.add((a * x).round(7))
            # Define a set of time points
            time_points = [0, 0.6e-6, 3.4e-6, 4e-6]

            # Define the strength of the transverse field Ω
            amplitude_min = 0
            amplitude_max = 10e6  # rad / sec

            # Define the strength of the detuning Δ
            Delta_initial = -20e6     # rad / sec
            Delta_final = Delta_final # Defined above

            # Define the total drive
            amplitude_values = [amplitude_min, amplitude_max, amplitude_max, amplitude_min]  # piecewise linear
            detuning_values = [Delta_initial, Delta_initial, Delta_final, Delta_final]  # piecewise linear
            phase_values = [0, 0, 0, 0]  # piecewise constant


            # Define the drive
            drive = get_drive(time_points, amplitude_values, detuning_values, phase_values)

            small_ahs_program = AnalogHamiltonianSimulation(
                register=small_register, 
                hamiltonian=drive
            )
            device = LocalSimulator("braket_ahs")
            small_ahs_run = device.run(small_ahs_program, shots=1000)
            result  = small_ahs_run.result()
            random_ext = np.random.randint(10000)
            save_result_json(f"result_{random_ext}.json",result)
            result_json = open_json(f"result_{random_ext}.json")
            IS_sizes,pp_bitstrings = analysis_MIS(G,result_json)
            ind, = np.where(IS_sizes==IS_sizes.max())
            ind_set = []
            for i in range(len(pp_bitstrings[ind[0]])):
                if pp_bitstrings[ind[0]][i] == 0:
                    ind_set.append(i)
            os.remove(f"result_{random_ext}.json")
        elif use_optimal == 'quera':
            task_arn = graph_io.check_if_graph_has_arn(G)
            if task_arn != None:
                task = AwsQuantumTask(arn = task_arn)
                random_ext = np.random.randint(10000)
                while True:
                    if task.state()=="COMPLETED":
                        save_result_json(f"result_{random_ext}.json",task.result())
                        break
                    else:
                        print('Task is pending. STATUS:',task.state())
                        time.sleep(60)
                result_json = open_json(f"result_{random_ext}.json")
                ind_sets = repair_quera_ind_sol(G, result_json)
                ind_set = []
                largest_size = 0
                for curr_set in ind_sets:
                    curr_set_pp = greedy_local_improv(G, curr_set)
                    if len(curr_set_pp) > largest_size:
                        ind_set = curr_set
                        largest_size = len(curr_set_pp)
                os.remove(f"result_{random_ext}.json")
            else:
                qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
                capabilities = qpu.properties.paradigm
                C6 = float(capabilities.rydberg.dict()['c6Coefficient'])
                unitdisk_radius,min_radius,max_radius = find_UDG_radius(pos,G)
                
                Delta_final = 20e6 # rad/sec
                blockade_radius = (C6/(Delta_final))**(1/6)
                a = blockade_radius / unitdisk_radius
                if a < 4e-6:
                    print("Forcing a to be 4e-6, hopefully it works...")
                    a = 4e-6
                register = AtomArrangement()
                register_pos = []
                for x in pos:
                    register.add([round(a * x[0],7), round(a*x[1],7)])
                    register_pos.append([round(a * x[0],7), round(a*x[1],7)])
                # Define a set of time points
                time_points = [0, 0.6e-6, 3.4e-6, 4e-6]

                # Define the strength of the transverse field Ω
                amplitude_min = 0
                amplitude_max = 10e6  # rad / sec

                # Define the strength of the detuning Δ
                Delta_initial = -20e6     # rad / sec
                Delta_final = Delta_final # Defined above

                # Define the total drive
                amplitude_values = [amplitude_min, amplitude_max, amplitude_max, amplitude_min]  # piecewise linear
                detuning_values = [Delta_initial, Delta_initial, Delta_final, Delta_final]  # piecewise linear
                phase_values = [0, 0, 0, 0]  # piecewise constant


                # Define the drive
                drive = get_drive(time_points, amplitude_values, detuning_values, phase_values)

                ahs_program = AnalogHamiltonianSimulation(
                    register=register, 
                    hamiltonian=drive
                )

                ind_set = []
                nshots = 200
                task = qpu.run(ahs_program, shots=nshots)
                task_arn = task.id
                task = AwsQuantumTask(arn = task_arn)
                random_ext = np.random.randint(10000)
                while True:
                    if task.state()=="COMPLETED":
                        save_result_json(f"result_{random_ext}.json",task.result())
                        break
                    else:
                        print('Task is pending. STATUS:',task.state())
                        time.sleep(60)
                result_json = open_json(f"result_{random_ext}.json")
                ind_sets = repair_quera_ind_sol(G, result_json)
                largest_size = 0
                for curr_set in ind_sets:
                    curr_set_pp = greedy_local_improv(G, curr_set)
                    if len(curr_set_pp) > largest_size:
                        ind_set = curr_set
                        largest_size = len(curr_set_pp)
                os.remove(f"result_{random_ext}.json")
                graph_io.save_quera_arn(nx.weisfeiler_lehman_graph_hash(G), task_arn)

        else:
            print('Solver not supported')
            exit(1)
    return ind_set

def find_graph_separator_kahip(G):
    random_ext = np.random.randint(10000)
    temp_input_name = f'separator_kahip_{random_ext}.graph'
    temp_input_file = open(temp_input_name, 'w')
    temp_input_file.write(f"{len(G.nodes)} {len(G.edges)}\n")
    for n in range(len(G.nodes)):
        for neighbor in sorted(list(G[n])):
            temp_input_file.write(f"{neighbor+1} ")
        temp_input_file.write('\n')
    temp_input_file.close()
    kahip_excutable = "/home/xu675/Research/KaHIP/deploy/node_separator"

    c_command = [kahip_excutable,temp_input_name]
    c_command.append(f"--output_filename=separator_kahip_{random_ext}.graph")
    c_command.append("--imbalance=5")
    c_command.append("--seed=42")

    subprocess.run(c_command ,stdout=subprocess.DEVNULL) 

    temp_out = open(f'separator_kahip_{random_ext}.graph', 'r')
    temp_out_lines = temp_out.read().splitlines()
    temp_out.close()
    A = []
    B = []
    S = []
    for i in range(len(temp_out_lines)):
        if int(temp_out_lines[i]) == 0:
            A.append(i)
        elif int(temp_out_lines[i]) == 1:
            B.append(i)
        elif int(temp_out_lines[i]) == 2:
            S.append(i)
    os.remove(f'separator_kahip_{random_ext}.graph')
    if len(A) + len(B) + len(S) != len(G.nodes):
        print("Separator error.. Check again")
        exit(1)
    # print(f"G:{len(G.nodes)}, S:{len(S)}, A:{len(A)}, B:{len(B)}")
    return A,B,S

def find_grpah_separator_grid(G, grid, cutoff):
    height = grid.shape[0]
    width = grid.shape[1]
    A = []
    B = []
    S = []
    if height > width:
        if height > cutoff:
            separator_row = int(np.floor(height / 2))
            grid_A = np.zeros((separator_row, width), dtype=int)
            grid_B = np.zeros((height-separator_row-1, width), dtype=int)
            for r in range(height):
                for c in range(width):
                    if r < separator_row:
                        if  grid[r,c] > -1:
                            A.append(grid[r,c])
                        grid_A[r,c] = grid[r,c]
                    elif r > separator_row:
                        if grid[r,c] > -1:
                            B.append(grid[r,c])
                        grid_B[r-separator_row-1,c] = grid[r,c]
                    elif r == separator_row and grid[r,c] > -1:
                        S.append(grid[r,c])
    else:
        if width > cutoff:
            separator_col = int(np.floor(width / 2))
            grid_A = np.zeros((height, separator_col), dtype=int)
            grid_B = np.zeros((height, width-separator_col-1), dtype=int)
            for c in range(width):
                for r in range(height):
                    if c < separator_col:
                        if  grid[r,c] > -1:
                            A.append(grid[r,c])
                        grid_A[r,c] = grid[r,c]
                    elif c > separator_col:
                        if grid[r,c] > -1:
                            B.append(grid[r,c])
                        grid_B[r,c-separator_col-1] = grid[r,c]
                    elif c == separator_col and grid[r,c] > -1:
                        S.append(grid[r,c])
    
    return A,B,S,grid_A,grid_B

def find_graph_separator(G):
    A = []
    B = []

    (edgecuts, parts) = metis.part_graph(G, 2, [0.5, 0.5], ubvec = [1.1])

    for i in range(len(G.nodes())):
        if parts[i] == 0:
            A.append(i)
        else:
            B.append(i)
    
    boundary_edges = list(nx.edge_boundary(G, B))

    boundary_subgraph = nx.Graph()
    boundary_subgraph.add_edges_from(boundary_edges)
    S = approximation.min_weighted_vertex_cover(boundary_subgraph)

    # boundary_nodes_A = list(nx.node_boundary(G, A))
    # boundary_nodes_B = list(nx.node_boundary(G, B))
    # diff_A = np.abs(len(A) - len(boundary_nodes_A) - len(B) )
    # diff_B = np.abs(len(B) - len(boundary_nodes_B) - len(A) )
    # if diff_A < diff_B:
    #     S = boundary_nodes_A
    # else:
    #     S = boundary_nodes_B

    G_copy = G.copy()
    for node in S:
        G_copy.remove_node(node)
    connected_components = list(nx.connected_components(G_copy))
    
    if nx.number_connected_components(G_copy) == 2:
        A = list(connected_components[0])
        B = list(connected_components[1])
    else:
        A = list(connected_components[0])
        B = list(connected_components[1])
        for remained_component in range(2, nx.number_connected_components(G_copy)):
            if len(A) < len(B):
                A += list(connected_components[remained_component])
            else:
                B += list(connected_components[remained_component])

    # print(f"S:{len(S)}, A:{len(A)}, B:{len(B)}")
    return A,B,S

def check_is_connected(G, u, S):
    for s in S:
        if G.has_edge(u,s):
            return True
    return False

def solve_mis_recursively(G, order, use_optimal, cutoff, depth, grid, pos=None):
    global graph_type
    global global_depth
    if grid.shape[0] <= cutoff and grid.shape[1] <= cutoff:
        ind_set = solve_local_mis(G, use_optimal=use_optimal, pos=pos)
        if global_depth < depth:
            global_depth = depth
    else:
        A,B,S,grid_A,grid_B = find_grpah_separator_grid(G, grid, cutoff)
        # node_to_community = {}
        # nodecolor = {}
        # nodesize = {}
        # for i in range(len(G.nodes)):
        #     if i in A:
        #         node_to_community[i] = 0
        #     elif i in B:
        #         node_to_community[i] = 1
        #     elif i in S:
        #         node_to_community[i] = 2
        
        # for i in range(len(G.nodes)):
        #     if  i in S:
        #         nodecolor[i] = 'white'
        #         nodesize[i] = 2.0
        #     elif i in A:
        #         nodecolor[i] = 'red'
        #         nodesize[i] = 1.0
        #     else:
        #         nodecolor[i] = 'blue'
        #         nodesize[i] = 1.0
        # print(nx.adjacency_matrix(G))
        # exit()
        # nodecolor = []
        # nodesize = []

        # red_dot = mlines.Line2D([], [], color='red', marker = '.', linestyle='None',
        #                   markersize=10, label='Partition 1')
        # blue_dot = mlines.Line2D([], [], color='blue', marker = '.', linestyle='None',
        #                   markersize=10, label='Partition 2')
        # gray_dot = mlines.Line2D([], [], color='gray', marker = '.', linestyle='None',
        #                   markersize=10, label='Separator')
        # for i in range(len(G.nodes)):
        #     if i in S:
        #         nodecolor.append('gray')
        #         nodesize.append(120)
        #     elif i in A:
        #         nodecolor.append('red')
        #         nodesize.append(50)
        #     else:
        #         nodecolor.append('blue')
        #         nodesize.append(50)

        # nx.draw(G, pos=nx.planar_layout(G), node_size = nodesize, node_color=nodecolor, edge_color='black', width =0.5)

        # Graph(list(G.edges), node_layout='spring', node_color = nodecolor, node_size = nodesize, edge_width=0.25)
        # # plt.legend(handles=[gray_dot, red_dot, blue_dot])
        # plt.savefig(f"depth_{depth}_{random.randint(1,1)}.pdf")
        # plt.clf()
        # exit()
            

        if order == 'top-down':
            exit(1)
        else:   
            #TODO: solve the MIS in and A and B, denote it as I_A and I_B, remove the vertices adjacent 
            # to I_A and I_B in S (also add vertices that are not covered?)and solve it
            if graph_type != 'king' and graph_type != 'geometric':
                subgraph_A = nx.induced_subgraph(G,A).copy()
                subgraph_A_nodes = list(subgraph_A.nodes())
                subgraph_A_relabelled = nx.convert_node_labels_to_integers(subgraph_A)
                subgraph_B = nx.induced_subgraph(G,B).copy()
                subgraph_B_nodes = list(subgraph_B.nodes())
                subgraph_B_relabelled = nx.convert_node_labels_to_integers(subgraph_B)
                ind_set_A = solve_mis_recursively(subgraph_A_relabelled, order, use_optimal,cutoff,depth+1)
                ind_set_A = greedy_local_improv(subgraph_A_relabelled, ind_set_A)
                ind_set_B = solve_mis_recursively(subgraph_B_relabelled, order, use_optimal,cutoff,depth+1)
                ind_set_B = greedy_local_improv(subgraph_B_relabelled, ind_set_B)
                mis_A = []
                for ind in ind_set_A:
                    mis_A.append(subgraph_A_nodes[ind])
                mis_B = []
                for ind in ind_set_B:
                    mis_B.append(subgraph_B_nodes[ind])

                S_temp = []
                for s in S:
                    if check_is_connected(G, s, A) == False and check_is_connected(G, s, B) == False:
                        S_temp.append(s)
                S = S_temp
                separator_subgraph = nx.induced_subgraph(G, S).copy()
                separator_subgraph_nodes = list(separator_subgraph.nodes())
                separator_subgraph_relablled = nx.convert_node_labels_to_integers(separator_subgraph)
                ind_set_S = solve_mis_recursively(separator_subgraph_relablled, order, use_optimal, cutoff,depth+1)
                ind_set_S = greedy_local_improv(separator_subgraph_relablled, ind_set_S)

                mis_S = []
                for ind in ind_set_S:
                    mis_S.append(separator_subgraph_nodes[ind])

                ind_set = mis_S + mis_A + mis_B
            else:
                def rescale_pos(original_pos, pos):
                    pos = pos.copy()
                    pos = pos.astype(float)
                    original_pos_x_max = np.max(original_pos[:,0])
                    original_pos_y_max = np.max(original_pos[:,1])
                    pos_x_max = np.max(pos[:,0])
                    pos_y_max = np.max(pos[:,1])
                    scalar = min(original_pos_y_max/pos_y_max, original_pos_x_max/pos_x_max)
                    print(scalar)
                    pos[:,0] *= scalar
                    pos[:,1] *= scalar
                    return pos

                subgraph_A = nx.induced_subgraph(G,A).copy()
                subgraph_A_nodes = list(subgraph_A.nodes())
                # subgraph_A_pos = pos[subgraph_A_nodes]
                # rescaled_subgraph_A_pos = rescale_pos(pos, subgraph_A_pos)
                subgraph_A_relabelled = nx.convert_node_labels_to_integers(subgraph_A)
                grid_A_relabelled = np.zeros((grid_A.shape[0], grid_A.shape[1]), dtype=int)
                grid_A_relabelled -= 1
                for i in range(grid_A.shape[0]):
                    for j in range(grid_A.shape[1]):
                        if grid_A[i,j] > -1:
                            grid_A_relabelled[i,j] = subgraph_A_nodes.index(grid_A[i,j])
                subgraph_A_pos = []
                for node in subgraph_A_nodes:
                    subgraph_A_pos.append([np.where(grid_A==node)[0][0],np.where(grid_A==node)[1][0]])
                subgraph_A_pos = np.asarray(subgraph_A_pos)

                subgraph_B = nx.induced_subgraph(G,B).copy()
                subgraph_B_nodes = list(subgraph_B.nodes())
                # subgraph_B_pos = pos[subgraph_B_nodes]
                subgraph_B_pos = []
                for node in subgraph_B_nodes:
                    subgraph_B_pos.append([np.where(grid_B==node)[0][0],np.where(grid_B==node)[1][0]])
                subgraph_B_pos = np.asarray(subgraph_B_pos)
                # rescaled_subgraph_B_pos = rescale_pos(pos, subgraph_B_pos)
                subgraph_B_relabelled = nx.convert_node_labels_to_integers(subgraph_B)
                grid_B_relabelled = np.zeros((grid_B.shape[0], grid_B.shape[1]), dtype=int)
                grid_B_relabelled -= 1
                for i in range(grid_B.shape[0]):
                    for j in range(grid_B.shape[1]):
                        if grid_B[i,j] > -1:
                            grid_B_relabelled[i,j] = subgraph_B_nodes.index(grid_B[i,j])

                ind_set_A = solve_mis_recursively(subgraph_A_relabelled, order, use_optimal,cutoff,depth+1, grid_A_relabelled, pos=subgraph_A_pos)
                ind_set_A = greedy_local_improv(subgraph_A_relabelled, ind_set_A)
                ind_set_B = solve_mis_recursively(subgraph_B_relabelled, order, use_optimal,cutoff,depth+1, grid_B_relabelled, pos=subgraph_B_pos)
                ind_set_B = greedy_local_improv(subgraph_B_relabelled, ind_set_B)
                mis_A = []
                for ind in ind_set_A:
                    mis_A.append(subgraph_A_nodes[ind])
                mis_B = []
                for ind in ind_set_B:
                    mis_B.append(subgraph_B_nodes[ind])

                S_temp = []
                for s in S:
                    if check_is_connected(G, s, A) == False and check_is_connected(G, s, B) == False:
                        S_temp.append(s)
                S = S_temp
                if not S:
                    ind_set = mis_A + mis_B
                else:
                    separator_subgraph = nx.induced_subgraph(G, S).copy()
                    ind_set_S = nx.maximal_independent_set(separator_subgraph)
                    separator_subgraph_nodes = list(separator_subgraph.nodes())
                    # separator_subgraph_pos = pos[separator_subgraph_nodes]
                    # # rescaled_separator_subgraph_pos = rescale_pos(pos, separator_subgraph_pos)
                    # separator_subgraph_relablled = nx.convert_node_labels_to_integers(separator_subgraph)
                    # ind_set_S = solve_mis_recursively(separator_subgraph_relablled, order, use_optimal, cutoff,depth+1,pos=separator_subgraph_pos)
                    # ind_set_S = greedy_local_improv(separator_subgraph_relablled, ind_set_S)

                    # mis_S = []
                    # for ind in ind_set_S:
                    #     mis_S.append(separator_subgraph_nodes[ind])

                    ind_set = ind_set_S + mis_A + mis_B

    return ind_set

def is_independent_set(G, indep_nodes):
    return len(G.subgraph(indep_nodes).edges) == 0

def greedy_local_improv(G, ind_set):
    for node in list(G.nodes()):
        if check_is_connected(G, node, ind_set) == False and node not in ind_set:
            ind_set.append(node)
    return ind_set

def run_luby_java(G):
    luby_path = "/home/xu675/Research/MIS-QAOA-Project/LubyMIS"
    random_ext = np.random.randint(10000)
    temp_input = open(f"luby_{random_ext}.in", 'w')

    temp_input.write(f"{len(G.nodes)}\n")
    for i in range(len(G.nodes)):
        temp_input.write(f"{i} ")
    temp_input.write("\n")

    A = nx.to_numpy_array(G)

    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            temp_input.write(f"{int(A[i,j])} ")
        temp_input.write("\n")
    
    temp_input.close()

    java_command = ['java', '-cp', luby_path, 'Luby', f"luby_{random_ext}.in", f"luby_{random_ext}.out"]

    # subprocess.run(java_command)
    subprocess.run(java_command, stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    
    temp_out = open(f"luby_{random_ext}.out", 'r')
    temp_out.readline()
    result = temp_out.readline()
    temp_out.close()

    os.remove(f"luby_{random_ext}.in")
    os.remove(f"luby_{random_ext}.out")

    return len(re.findall(r'\d+', result))

def run_kamis(G):
    kamis_path = '/home/xu675/Research/KaMIS/deploy/redumis'
    random_ext = np.random.randint(10000)
    temp_input_name = f'kamis_{random_ext}.graph'
    temp_input_file = open(temp_input_name, 'w')
    temp_input_file.write(f"{len(G.nodes)} {len(G.edges)}\n")
    for n in range(len(G.nodes)):
        for neighbor in sorted(list(G[n])):
            temp_input_file.write(f"{neighbor+1} ")
        temp_input_file.write('\n')
    temp_input_file.close()

    commands = [kamis_path, f'kamis_{random_ext}.graph', f'--output=kamis_{random_ext}.out', '--seed=42']
    subprocess.run(commands ,stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL) 
    output = open(f'kamis_{random_ext}.out', 'r')

    ind_set = []

    ind_set_size = 0
    line_count = 0
    for line in output:
        if int(line) == 1:
            ind_set_size += 1
            ind_set.append(line_count)
        line_count += 1
    
    output.close()
    os.remove(f'kamis_{random_ext}.graph')
    os.remove(f'kamis_{random_ext}.out')
    
    return ind_set_size, ind_set




def read_leda_graph(num_nodes):
    filename = f"/home/xu675/Research/MIS-QAOA-Project/MIS-Divide-and-Conquer/planar{os.sep}{num_nodes}.leda"
    if Path(filename).is_file() == False:
        kahip_excutable = "/home/xu675/randomgraph/randomgraph"
        c_command = [kahip_excutable]
        c_command.append(str(num_nodes))
        c_command.append("-o")
        c_command.append(filename)
        subprocess.run(c_command ,stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL) 

    inputfile = open(filename, 'r')
    for i in range(3):
        inputfile.readline()
    if num_nodes != int(inputfile.readline()):
        print("Input Graph does not match desired node count..")
        exit(1)
    for i in range(num_nodes):
        inputfile.readline()
    num_edges = int(inputfile.readline())
    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))

    for i in range(num_edges):
        line = inputfile.readline()
        strings = line.split()
        G.add_edge(int(strings[0])-1, int(strings[1])-1)

    if nx.is_planar(G) == False:
        print("The grpah is not planar?")
        exit(1)
    
    print(f"The graph has {num_nodes} nodes and {num_edges} edges.")

    return G

def drop_weights(G):
    '''Drop the weights from a networkx weighted graph.'''
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs.pop('weight', None)

def remove_self_loop(G):
    for edge in G.edges:
        if edge[0] == edge[1]:
            G.remove_edge(edge[0], edge[1])


def generate_random_pos(n, minimum_dist):
    def distance_between_points(a,b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    pos = [[0.5, 0.5]]
    while len(pos) < n:        
        proposed_point = np.random.rand(2)
        add_flag = True
        for exisiting_point in pos:
            if distance_between_points(exisiting_point, proposed_point) < minimum_dist:
                add_flag = False
                break
        if add_flag:
            pos.append([proposed_point[0], proposed_point[1]])
    return np.asarray(pos)

if __name__ == '__main__':
    # id = '9f91c90b-9525-43c2-a4a8-fecb82207259'
    # task_arn = f'arn:aws:braket:us-east-1:632449754113:quantum-task/{id}'
    # task = AwsQuantumTask(arn = task_arn)
    # random_ext = np.random.randint(10000)
    # while True:
    #     if task.state()=="COMPLETED":
    #         save_result_json(f"result_{random_ext}.json",task.result())
    #         break
    #     else:
    #         print('Task is pending. STATUS:',task.state())
    #         time.sleep(60)
    # result_json = open_json(f"result_{random_ext}.json")
    # check = check_quera_results(result_json)
    # if check == False:
    #     print("Wrong Init")
    # else:
    #     print('OK')
    # os.remove(f"result_{random_ext}.json")
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', metavar='x',nargs='?', type=int, help='Number of nodes')
    parser.add_argument('-y', metavar='y',nargs='?', type=int, help='Number of nodes')
    parser.add_argument('-opt', metavar='use_optimal_mis_solver',type=str, help='1 for using optimal solver and 0 for using QAOA')
    parser.add_argument('-cutoff', metavar='cutoff_for_local_solver',type=int, nargs='?', default = 16, help='graphs below this number of nodes are solved using local MIS solver')
    parser.add_argument('-filling', metavar='filling',type=float, help='graphs below this number of nodes are solved using local MIS solver')
    args = parser.parse_args()

    
    dim_x = args.x
    dim_y = args.y
    use_optimal = args.opt
    cutoff = args.cutoff
    filling = args.filling

    graph_type = 'king'

    if graph_type == 'king':
        pos, G = kings_graph(dim_x,dim_y,filling,seed = 1)
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
        # visualize_graph(ax,G,pos)
        # plt.tight_layout()
        # plt.savefig('kings_example.pdf')
        # exit()
        grid = np.zeros((dim_x, dim_y), dtype=int)
        grid -= 1
        for i in range(len(G.nodes())):
            grid[pos[i,0], pos[i,1]] = i
    elif graph_type == 'geometric':
        pos = generate_random_pos(dim_x, 0.1)
        generated_pos_dict = {}
        for i in range(dim_x):
            generated_pos_dict[i] = pos[i]
        G = nx.random_geometric_graph(dim_x, 0.3, pos=generated_pos_dict)
        # pos_dict = nx.get_node_attributes(G, 'pos')
        # pos = []
        # for i in range(len(G.nodes())):
        #     pos.append(pos_dict[i])
        # pos = np.asarray(pos)

    print(f"The graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    print(f"The Graph is planar: {nx.is_planar(G)}")
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    print(f"Highest degree in the graph is {degrees[0][1]}. Average Degree is {np.average(degrees,axis=0)[1]}. Degree std is {np.std(degrees,axis=0)[1]}")

    kamis_result, ind_set = run_kamis(G)
    if is_independent_set(G, ind_set) == False:
            print("Kamis Set is not independent! Chekc your code...")
            exit(1)
    print('-------------------------------------------')
    print("KaMIS size:", kamis_result)

    luby_runs = 10
    best_luby_result = 0
    for i in range(10):
        luby_mis = run_luby_java(G)
        if best_luby_result < luby_mis:
            best_luby_result = luby_mis
    
    print(f"Best Luby MIS size out of {luby_runs} runs is {best_luby_result}")
    print('-------------------------------------------')

    for order in ['bottom-up']:
        global global_depth
        global_depth = 0
        print(f"Current direction is {order}.")
        if graph_type != 'king' and graph_type != 'geometric':
            ind_set = solve_mis_recursively(G, order, use_optimal, cutoff, 0)
        else:
            ind_set = solve_mis_recursively(G, order, use_optimal, cutoff, 0, grid, pos=pos)
        if is_independent_set(G, ind_set) == False:
            print("Set is not independent! Chekc your code...")
            exit(1)
        print(f"D-C MIS size before greedy improvement is {len(ind_set)} with depths of {global_depth+1} using {use_optimal} solver")
        improved_ind_set = greedy_local_improv(G, ind_set)
        if is_independent_set(G, improved_ind_set) == False:
            print("Imrpoved set is not independent! Chekc your code...")
            exit(1)

        print(f"D-C MIS size after greedy improvement is {len(improved_ind_set)} using {use_optimal} solver")


    
    

