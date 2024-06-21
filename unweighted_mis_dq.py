import networkx as nx
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import itertools
import subprocess
import os
import argparse
from pathlib import Path
import re
from scipy.io import mmread
import random
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.samplers import SteepestDescentSolver
import pickle
import neal
import dimod
from minorminer import find_embedding
from utils import graph_io

def maximum_weighted_independent_set_qubo(G, qubo_scalar, penalty_multiplier=2.0):
    if not G:
        return {}
    cost = {}
    for i in range(len(G.nodes)):
        cost[i] = G.nodes[i]['weight']
    Q = {(node, node): -cost[node] for node in G}
    for (u,v) in G.edges():
        Q[(u,v)] = penalty_multiplier * min(G.nodes[u]['weight'],G.nodes[v]['weight'])
    Q = {x:y for x,y in Q.items() if y!=0}

    return Q

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
    
def parse_dwave_result(sample):
    sol = []
    for i in sample:
        if sample[i] == 1:
            sol.append(int(i))
    return sol


def solve_local_mis(G, local_solver):
    # pickle.dump(G,open('cegb3024.pickle','wb'))
    # exit()
    global dwave_results_folder
    annealing_time = 50
    num_qpu_calls = 10
    num_reads_per_qpu_call = 100
    sample_total_count = 100
    

    if len(G.nodes) == 1:
        ind_set = G.nodes()
    elif len(G.nodes) == 0:
        ind_set = []
    else:
        if local_solver == 'dwave':
            total_samplesets = graph_io.check_if_graph_already_solved(G, dwave_results_folder, annealing_time, num_reads_per_qpu_call*num_qpu_calls)
            if total_samplesets == None:
                Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=2.0)
                qpu_advantage = DWaveSampler(solver='Advantage_system6.4')
                embedding = find_embedding(G, qpu_advantage.properties["couplers"], chainlength_patience=50, max_no_improvement=50 )
                sampler = FixedEmbeddingComposite(qpu_advantage, embedding=embedding)

                samplesets = []
                for _ in range(num_qpu_calls):
                    sampleset = sampler.sample_qubo(Q, num_reads=num_reads_per_qpu_call,annealing_time=annealing_time)
                    samplesets.append(sampleset)
                
                total_samplesets = dimod.concatenate(samplesets)
                best_sample = None
                best_sample_indset_size = 0
                sample_count = 0
                for sample,energy in total_samplesets.data(fields=['sample','energy'], sorted_by='energy'):
                    curr_ind_set = parse_dwave_result(sample)
                    curr_ind_set = repair_dwave_ind_sol(G, curr_ind_set)
                    curr_ind_set = greedy_local_improv(G, curr_ind_set)
                    if get_ind_size_weight(G, curr_ind_set) > best_sample_indset_size:
                        best_sample = sample
                        best_sample_indset_size = get_ind_size_weight(G, curr_ind_set)
                    sample_count += 1
                    if sample_count == sample_total_count:
                        break

                graph_io.save_sampleset(total_samplesets,nx.weisfeiler_lehman_graph_hash(G, node_attr='weight'),dwave_results_folder, annealing_time, num_reads_per_qpu_call*num_qpu_calls)
                ind_set = parse_dwave_result(best_sample)
                ind_set = repair_dwave_ind_sol(G, ind_set)
            else:
                best_sample = None
                best_sample_indset_size = 0
                sample_count = 0
                for sample,energy in total_samplesets.data(fields=['sample','energy'], sorted_by='energy'):
                    curr_ind_set = parse_dwave_result(sample)
                    curr_ind_set = repair_dwave_ind_sol(G, curr_ind_set)
                    curr_ind_set = greedy_local_improv(G, curr_ind_set)
                    if get_ind_size_weight(G, curr_ind_set) > best_sample_indset_size:
                        best_sample = sample
                        best_sample_indset_size = get_ind_size_weight(G, curr_ind_set)
                    sample_count += 1
                    if sample_count == sample_total_count:
                        break
                ind_set = parse_dwave_result(best_sample)
                ind_set = repair_dwave_ind_sol(G, ind_set)
            
        elif local_solver == 'kamis':
            ind_size, ind_set = run_kamis(G)
        elif local_solver == 'sa':
            Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=2.0)
            sampler = neal.SimulatedAnnealingSampler()
            total_samplesets = sampler.sample_qubo(Q, num_reads=1000)
            best_sample = None
            best_sample_indset_size = 0
            sample_count = 0
            for sample,energy in total_samplesets.data(fields=['sample','energy'], sorted_by='energy'):
                curr_ind_set = parse_dwave_result(sample)
                curr_ind_set = repair_dwave_ind_sol(G, curr_ind_set)
                curr_ind_set = greedy_local_improv(G, curr_ind_set)
                if get_ind_size_weight(G, curr_ind_set) > best_sample_indset_size:
                    best_sample = sample
                    best_sample_indset_size = get_ind_size_weight(G, curr_ind_set)
                sample_count += 1
                if sample_count == sample_total_count:
                    break
            ind_set = parse_dwave_result(best_sample)
            ind_set = repair_dwave_ind_sol(G, ind_set)
        elif local_solver == 'dwave_greedy':
            Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=2.0)
            total_samplesets = graph_io.check_if_graph_already_solved(G, dwave_results_folder, annealing_time, num_reads_per_qpu_call*num_qpu_calls)
            if total_samplesets == None:
                print("Run this local_solver='dwave' first")
                exit(1)
                Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=1.1)
                sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system6.4'))
                sampleset = sampler.sample_qubo(Q, num_reads=100,annealing_time=50)
                save_sampleset(sampleset,nx.weisfeiler_lehman_graph_hash(G, node_attr='weight'),dwave_results_folder, annealing_time, num_reads)
            solver_greedy = SteepestDescentSolver()
            sampleset_pp = solver_greedy.sample_qubo(Q, initial_states=total_samplesets)  

            best_sample = None
            best_sample_indset_size = 0
            sample_count = 0
            for sample,energy in sampleset_pp.data(fields=['sample','energy'], sorted_by='energy'):
                curr_ind_set = parse_dwave_result(sample)
                curr_ind_set = repair_dwave_ind_sol(G, curr_ind_set)
                curr_ind_set = greedy_local_improv(G, curr_ind_set)
                if get_ind_size_weight(G, curr_ind_set) > best_sample_indset_size:
                    best_sample = sample
                    best_sample_indset_size = get_ind_size_weight(G, curr_ind_set)
                sample_count += 1
                if sample_count == sample_total_count:
                    break

            ind_set = parse_dwave_result(best_sample)
            ind_set = repair_dwave_ind_sol(G, ind_set)
        elif local_solver == 'exact':
            exact_folder = './exact_weighted_results'
            sampleset = graph_io.check_if_graph_already_solved(G, exact_folder, 0, 0)
            if sampleset == None:
                Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=2.0)
                sampler = dimod.ExactSolver()
                sampleset = sampler.sample_qubo(Q)
                graph_io.save_sampleset(sampleset,nx.weisfeiler_lehman_graph_hash(G, node_attr='weight'),exact_folder, 0, 0)
                ind_set = parse_dwave_result(sampleset.first.sample)
                if not is_independent_set(G, ind_set):
                    print("Something wrong with the exact solver")
                    exit(1)
            else:
                ind_set = parse_dwave_result(sampleset.first.sample)
            
    return ind_set


def find_graph_separator_kahip(G):

    separators = graph_io.check_if_graph_already_separated(G)
    if separators == None:
        random_ext = np.random.randint(10000)
        temp_input_name = f'separator_kahip_{random_ext}.graph'
        temp_input_file = open(temp_input_name, 'w')
        # temp_input_file.write(f"{len(G.nodes)} {len(G.edges)} 10\n")
        temp_input_file.write(f"{len(G.nodes)} {len(G.edges)}\n")
        for n in range(len(G.nodes)):
            # temp_input_file.write(f"{int(G.nodes[n]['weight'])} ")
            for neighbor in sorted(list(G[n])):
                temp_input_file.write(f"{neighbor+1} ")
            temp_input_file.write('\n')
        temp_input_file.close()
        kahip_excutable = "/depot/arnabb/data/hanjing/Research/KaHIP/deploy/node_separator"

        c_command = [kahip_excutable,temp_input_name]
        c_command.append(f"--output_filename=separator_kahip_{random_ext}.out")
        c_command.append("--imbalance=5")
        c_command.append("--seed=42")

        subprocess.run(c_command ,stdout=subprocess.DEVNULL) 

        temp_out = open(f'separator_kahip_{random_ext}.out', 'r')
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
        os.remove(f'separator_kahip_{random_ext}.out')
        if len(A) + len(B) + len(S) != len(G.nodes):
            print("Separator error.. Check again")
            exit(1)
        graph_io.save_separator((A,B,S),nx.weisfeiler_lehman_graph_hash(G, node_attr='weight') )
    else:
        A = separators[0]
        B = separators[1]
        S = separators[2]
        
        
    # print(f"G:{len(G.nodes)}, S:{len(S)}, A:{len(A)}, B:{len(B)}")
    return A,B,S

def check_is_connected(G, u, S):
    for s in S:
        if G.has_edge(u,s):
            return True
    return False

def solve_mis_recursively(G, order, local_solver, cutoff, depth):
    global global_depth
    if len(G.nodes()) <= 15:
        ind_set = solve_local_mis(G, local_solver='exact')
        if global_depth < depth:
            global_depth = depth
    elif len(G.nodes()) <= cutoff:
        ind_set = solve_local_mis(G, local_solver=local_solver)
        if global_depth < depth:
            global_depth = depth
    else:
        A,B,S = find_graph_separator_kahip(G)
        if min(A) > min(B):
            temp = A
            A = B
            B = temp

        if order == 'top-down':
            #TODO: solve the MIS in S, denote it as I_S, remove vertices adjacent to I_S in A and B and solve them
            separator_subgraph = nx.induced_subgraph(G, S).copy()
            separator_subgraph_nodes = list(separator_subgraph.nodes())
            separator_subgraph_relablled = nx.convert_node_labels_to_integers(separator_subgraph)
            ind_set_S = solve_mis_recursively(separator_subgraph_relablled, order, local_solver, cutoff, depth+1)
            # if use_greedy_improv:
            ind_set_S = greedy_local_improv(separator_subgraph_relablled, ind_set_S)

            mis_S = []
            for ind in ind_set_S:
                mis_S.append(separator_subgraph_nodes[ind])
            A_temp = []
            B_temp = []
            for a in A:
                if not check_is_connected(G, a, mis_S):
                    A_temp.append(a)
            for b in B:
                if not check_is_connected(G, b, mis_S):
                    B_temp.append(b)
            A = A_temp
            B = B_temp

            subgraph_A = nx.induced_subgraph(G,A).copy()
            subgraph_A_nodes = list(subgraph_A.nodes())
            subgraph_A_relabelled = nx.convert_node_labels_to_integers(subgraph_A)
            subgraph_B = nx.induced_subgraph(G,B).copy()
            subgraph_B_nodes = list(subgraph_B.nodes())
            subgraph_B_relabelled = nx.convert_node_labels_to_integers(subgraph_B)
            ind_set_A = solve_mis_recursively(subgraph_A_relabelled, order, local_solver, cutoff, depth+1)
            ind_set_A = greedy_local_improv(subgraph_A_relabelled, ind_set_A)

            ind_set_B = solve_mis_recursively(subgraph_B_relabelled, order, local_solver, cutoff, depth+1)
            ind_set_B = greedy_local_improv(subgraph_B_relabelled, ind_set_B)
            mis_A = []
            for ind in ind_set_A:
                mis_A.append(subgraph_A_nodes[ind])
            mis_B = []
            for ind in ind_set_B:
                mis_B.append(subgraph_B_nodes[ind])
            ind_set = mis_S + mis_A + mis_B
            ind_set = greedy_local_improv(G, ind_set)

            # draw_color = []
            # for n in range(len(G.nodes)):
            #     if n in mis_S:
            #         draw_color.append('red')
            #     elif n in mis_A:
            #         draw_color.append('blue')
            #     elif n in mis_B:
            #         draw_color.append('green')
            #     else:
            #         draw_color.append('white')
            # plt.figure(figsize=(10,10))
            # nx.draw(G, with_labels = True, node_color = draw_color)
            # plt.savefig("is_graph.png")
        else:   
            #TODO: solve the MIS in and A and B, denote it as I_A and I_B, remove the vertices adjacent 
            # to I_A and I_B in S (also add vertices that are not covered?)and solve it
            subgraph_A = nx.induced_subgraph(G,A).copy()
            subgraph_A_nodes = list(subgraph_A.nodes())
            subgraph_A_relabelled = nx.convert_node_labels_to_integers(subgraph_A)
            subgraph_B = nx.induced_subgraph(G,B).copy()
            subgraph_B_nodes = list(subgraph_B.nodes())
            subgraph_B_relabelled = nx.convert_node_labels_to_integers(subgraph_B)
            ind_set_A = solve_mis_recursively(subgraph_A_relabelled, order, local_solver,cutoff,depth+1)
            ind_set_A = greedy_local_improv(subgraph_A_relabelled, ind_set_A)
            ind_set_B = solve_mis_recursively(subgraph_B_relabelled, order, local_solver,cutoff,depth+1)
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
            ind_set_S = solve_mis_recursively(separator_subgraph_relablled, order, local_solver, cutoff,depth+1)
            ind_set_S = greedy_local_improv(separator_subgraph_relablled, ind_set_S)

            mis_S = []
            for ind in ind_set_S:
                mis_S.append(separator_subgraph_nodes[ind])

            ind_set = mis_S + mis_A + mis_B

    return ind_set

def is_independent_set(G, ind_set):
    return len(G.subgraph(ind_set).edges) == 0

def is_maximal(G, ind_set):
    for node in G.nodes():
       if  check_is_connected(G, node, ind_set) == False and node not in ind_set:
           return False
    return True

def greedy_local_improv(G, ind_set):
    for node in sorted(G.nodes(), key=lambda n: G.nodes[n]['weight'], reverse=True):
        if check_is_connected(G, node, ind_set) == False and node not in ind_set:
            ind_set.append(node)
    return ind_set

def run_luby_java(G):
    random_ext = np.random.randint(10000)
    luby_path = "/depot/arnabb/data/hanjing/Research/MIS-QAOA-Project/LubyMIS"
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

    java_command = ['java', '-cp', luby_path, 'Luby', f'luby_{random_ext}.in', f'luby_{random_ext}.out']

    # subprocess.run(java_command)
    subprocess.run(java_command, stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    
    temp_out = open(f"luby_{random_ext}.out", 'r')
    temp_out.readline()
    result = temp_out.readline()
    temp_out.close()

    luby_set = list(map(int, list(re.findall(r'\d+', result))))
    if is_independent_set(G, luby_set) == False:
        print("Luby set is not independent! Chekc your code...")
        exit(1)

    if is_maximal(G, luby_set) == False:
        print("Luby's result is not maxmal")
        exit(1)

    os.remove(f'luby_{random_ext}.in')
    os.remove(f'luby_{random_ext}.out')
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

    commands = [kamis_path, f'kamis_{random_ext}.graph', f'--output=kamis_{random_ext}.out']
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

def assign_random_weights_to_graph(G, seed=42):
    for i in range(len(G.nodes)):
        G.nodes[i]['weight']=1
    return G
    
def get_ind_size_weight(G, ind_set):
    total_weights = 0
    for n in ind_set:
        total_weights += G.nodes[n]['weight']
    return total_weights



if __name__ == '__main__':
    global dwave_results_folder
    dwave_results_folder = './dwave_unweighted_results'

    parser = argparse.ArgumentParser()
    parser.add_argument('-local_solver', metavar='local_solver',type=str, help='1 for using optimal solver and 0 for using QAOA')
    parser.add_argument('-cutoff', metavar='cutoff_for_local_solver',type=int, nargs='?', default = 15, help='graphs below this number of nodes are solved using local MIS solver')
    parser.add_argument('-mtx', metavar='mtx_file_path',type=str, nargs='?', default = None, help='MTX')
    args = parser.parse_args()

    local_solver = args.local_solver
    cutoff = args.cutoff
    mtx_filename = args.mtx

    if mtx_filename == None:
        print("Input a graph name ends with .mtx")
        exit(1)
    else:
        a = mmread(f'/home/xu675/Research/MIS-QAOA-Project/MIS-Divide-and-Conquer/planar{os.sep}{mtx_filename}.mtx')
        G = nx.Graph(a)
        drop_weights(G)
        remove_self_loop(G)
        print(f"The graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
        G = assign_random_weights_to_graph(G)
    
    print(f"The Graph is planar: {nx.is_planar(G)}")
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    print(f"Highest degree in the graph is {degrees[0][1]}. Average Degree is {np.average(degrees,axis=0)[1]}. Degree std is {np.std(degrees,axis=0)[1]}")
    print('-------------------------------------------')

    kamis_runs = 10
    best_kamis_result = 0
    for _ in range(10):
        redumis_size, ind_set = run_kamis(G)
        if is_independent_set(G, ind_set) == False:
                print("Kamis set is not independent! Chekc your code...")
                exit(1)
        if best_kamis_result < redumis_size:
            best_kamis_result = redumis_size

    print(f"Best Kamis size out of {kamis_runs} runs is {best_kamis_result}")

    luby_runs = 10
    best_luby_result = 0
    for _ in range(10):
        luby_mis = run_luby_java(G)
        if best_luby_result < luby_mis:
            best_luby_result = luby_mis
    
    print(f"Best Luby MIS size out of {luby_runs} runs is {best_luby_result}")
    
    print('-------------------------------------------')
    
    for order in ['bottom-up']:
        global global_depth
        global_depth = 0
        print(f"Current direction is {order}.")
        ind_set = solve_mis_recursively(G, order, local_solver, cutoff, 0)
        if is_independent_set(G, ind_set) == False:
            print("Set is not independent! Chekc your code...")
            exit(1)

        print(f"D-C MIS size before greedy improvement is {round(get_ind_size_weight(G, ind_set),2)} with depths of {global_depth+1} using {local_solver}")
 
        improved_ind_set = greedy_local_improv(G, ind_set)
        if is_independent_set(G, improved_ind_set) == False:
            print("Imrpoved set is not independent! Chekc your code...")
            exit(1)
        print(f"D-C MIS size after greedy improvement is {get_ind_size_weight(G, improved_ind_set)} using {local_solver}")
    
