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
import time

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
    global total_prostprocess_time
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
            global total_mis_qpu_tasks
            total_mis_qpu_tasks += 1
            total_samplesets = graph_io.check_if_graph_already_solved(G, dwave_results_folder, annealing_time, num_reads_per_qpu_call*num_qpu_calls)
            if total_samplesets == None:
                global total_qpu_access_time
                global total_embedding_time
                Q = maximum_weighted_independent_set_qubo(G,10, penalty_multiplier=2.0)
                qpu_advantage = DWaveSampler(solver='Advantage_system6.4')
                starttime = time.time()
                embedding = find_embedding(G, qpu_advantage.properties["couplers"])
                total_embedding_time += time.time() - starttime
                sampler = FixedEmbeddingComposite(qpu_advantage, embedding=embedding)

                samplesets = []
                for _ in range(num_qpu_calls):
                    sampleset = sampler.sample_qubo(Q, num_reads=num_reads_per_qpu_call,annealing_time=annealing_time)
                    total_qpu_access_time += sampleset.info["timing"]["qpu_access_time"]
                    samplesets.append(sampleset)
                
                total_samplesets = dimod.concatenate(samplesets)
                best_sample = None
                best_sample_indset_size = 0
                sample_count = 0
                starttime = time.time()
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
                total_prostprocess_time += time.time() - starttime
                graph_io.save_sampleset(total_samplesets,nx.weisfeiler_lehman_graph_hash(G, node_attr='weight'),dwave_results_folder, annealing_time, num_reads_per_qpu_call*num_qpu_calls)
            else:
                best_sample = None
                best_sample_indset_size = 0
                sample_count = 0
                starttime = time.time()
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
                total_prostprocess_time += time.time() - starttime
            
        elif local_solver == 'kamis':
            ind_size, ind_set = run_kamis(G, 'weighted_local_search')
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
        global total_kahip_time
        starttime = time.time()
        A,B,S = find_graph_separator_kahip(G)
        total_kahip_time += time.time() - starttime

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

def is_independent_set(G, indep_nodes):
    return len(G.subgraph(indep_nodes).edges) == 0

def greedy_local_improv(G, ind_set):
    for node in sorted(G.nodes(), key=lambda n: G.nodes[n]['weight'], reverse=True):
        if check_is_connected(G, node, ind_set) == False and node not in ind_set:
            ind_set.append(node)
    return ind_set

def run_luby_java(G):
    random_ext = np.random.randint(10000)
    luby_path = "/depot/arnabb/data/hanjing/Research/MIS-QAOA-Project/LubyMIS"
    temp_input = open(f"temp_{random_ext}.in", 'w')

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

    java_command = ['java', '-cp', luby_path, 'Luby', f'temp_{random_ext}.in', 'temp_{random_ext}.out']

    # subprocess.run(java_command)
    subprocess.run(java_command, stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    
    temp_out = open("temp.out", 'r')
    temp_out.readline()
    result = temp_out.readline()
    temp_out.close()

    return len(re.findall(r'\d+', result))

def run_kamis(G, method):
    random_ext = np.random.randint(10000)
    kamis_path = f'/home/xu675/Research/KaMIS/deploy/{method}'
    temp_input_name = f'kamis_{random_ext}.graph'
    temp_input_file = open(temp_input_name, 'w')
    temp_input_file.write(f"{len(G.nodes)} {len(G.edges)} 10\n")
    for n in range(len(G.nodes)):
        temp_input_file.write(f"{int(G.nodes[n]['weight'])} ")
        for neighbor in sorted(list(G[n])):
            temp_input_file.write(f"{neighbor+1} ")
        temp_input_file.write('\n')
    temp_input_file.close()

    commands = [kamis_path, f'kamis_{random_ext}.graph', f'--output=kamis_{random_ext}.out', '--time_limit=300']
    subprocess.run(commands ,stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL) 
    output = open(f'kamis_{random_ext}.out', 'r')

    counter = 0
    ind_set_size = 0
    ind_set = []
    for line in output:
        if int(line) == 1:
            ind_set_size += G.nodes[counter]['weight']
            ind_set.append(counter)
        counter += 1
    output.close()

    os.remove(temp_input_name)
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
    # rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    # weights = rs.random_sample( len(G.nodes))
    random.seed(seed)
    weights = list(range(1,100))
    for i in range(len(G.nodes)):
        # G.nodes[i]['weight']=np.ceil(weights[i] * 10)
        G.nodes[i]['weight']=random.choice(weights)
    return G
    
def get_ind_size_weight(G, ind_set):
    total_weights = 0
    for n in ind_set:
        total_weights += G.nodes[n]['weight']
    return total_weights



if __name__ == '__main__':
    # G = pickle.load(open('dwave_parameter_test/man_5976.pickle','rb'))
    # ind_size, ind_set = run_kamis(G, 'weighted_local_search')
    # if is_independent_set(G, ind_set) == False:
    #     print("Something's Wrong with Kamis!")
    #     exit(1)
    # print(ind_set)
    # print(f'Kamis: {ind_size}')
    # exit()

    # qpu_advantage = DWaveSampler(solver='Advantage_system6.4')
    # Q = maximum_weighted_independent_set_qubo(G, 10, penalty_multiplier=1.1)
    # embedding = find_embedding(G, qpu_advantage.properties["couplers"], chainlength_patience=50,max_no_improvement=50 )
    # sampler = FixedEmbeddingComposite(DWaveSampler(solver='Advantage_system6.4'), embedding=embedding)
    # # sampler = neal.SimulatedAnnealingSampler()

    # samplesets = []
    # for _ in range(5):
    #     sampleset = sampler.sample_qubo(Q, num_reads=100,annealing_time=50)
    #     samplesets.append(sampleset)

    # total_samplesets = dimod.concatenate(samplesets)

    # sample_used = 100

    # best_sample = None
    # best_sample_indset_size = 0
    # for sample,energy in sampleset.data(fields=['sample','energy'], sorted_by='energy'):
    #     curr_ind_set = parse_dwave_result(sample)
    #     curr_ind_set = repair_dwave_ind_sol(G, curr_ind_set)
    #     curr_ind_set = greedy_local_improv(G, curr_ind_set)
    #     if get_ind_size_weight(G, curr_ind_set) > best_sample_indset_size:
    #         best_sample = sample
    #         best_sample_indset_size = get_ind_size_weight(G, curr_ind_set)

    # print('------------------------------')
    # best_ind_set = parse_dwave_result(best_sample)
    # best_ind_set = repair_dwave_ind_sol(G, best_ind_set)
    # best_ind_set = greedy_local_improv(G, best_ind_set)
    # print(f'DWave: {get_ind_size_weight(G, best_ind_set)}')
        
        
    # # solver_greedy = SteepestDescentSolver()
    # # sampleset_pp = solver_greedy.sample_qubo(Q, initial_states=sampleset)  

    # # pickle.dump(sampleset, open('sampleset.pickle', 'wb'))
    # # sampleset = pickle.load(open('sampleset.pickle', 'rb'))

    # exit()

    global dwave_results_folder
    dwave_results_folder = './dwave_weighted_results'

    global total_qpu_access_time
    total_qpu_access_time = 0

    global total_mis_qpu_tasks
    total_mis_qpu_tasks = 0

    global total_kahip_time
    total_kahip_time = 0

    global total_embedding_time
    total_embedding_time = 0   

    global total_prostprocess_time  
    total_prostprocess_time = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', metavar='num_nodes',type=int, help='Number of nodes')
    # parser.add_argument('-dir', metavar='direction',type=str, help='top-down or bottom-up')
    parser.add_argument('-local_solver', metavar='local_solver',type=str, help='1 for using optimal solver and 0 for using QAOA')
    parser.add_argument('-cutoff', metavar='cutoff_for_local_solver',type=int, nargs='?', default = 15, help='graphs below this number of nodes are solved using local MIS solver')
    parser.add_argument('-mtx', metavar='mtx_file_path',type=str, nargs='?', default = None, help='MTX')
    args = parser.parse_args()

    num_nodes = args.n
    # order = args.dir
    local_solver = args.local_solver
    cutoff = args.cutoff
    mtx_filename = args.mtx

    if mtx_filename == None:
        G = read_leda_graph(num_nodes)
        G = assign_random_weights_to_graph(G)
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
    
    # weighted_branch_reduce_kamis_result, ind_set = run_kamis(G, 'weighted_branch_reduce')
    # if is_independent_set(G, ind_set) == False:
    #         print("Set is not independent! Chekc your code...")
    #         exit(1)
    starttime = time.time()
    weighted_local_search_kamis_result, ind_set = run_kamis(G, 'weighted_local_search')
    if is_independent_set(G, ind_set) == False:
            print("Set is not independent! Chekc your code...")
            exit(1)
    # print("weighted_branch_reduce size:", round(weighted_branch_reduce_kamis_result,2))
    print("weighted_local_search size:", round(weighted_local_search_kamis_result, 2))
    
    kamis_time = time.time() - starttime
    print('-------------------------------------------')
    
    for order in ['bottom-up']:
        global global_depth
        global_depth = 0
        print(f"Current direction is {order}.")
        starttime = time.time()
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
        print('-------------------------------------------')
        print(f"Kamis used {kamis_time} seconds")
        print(f"Total number of MIS subproblem is {total_mis_qpu_tasks}")
        print(f"Total Kahip separator time is {total_kahip_time} seconds")
        print(f"Total minorminer time is {total_embedding_time} seconds")
        print(f"Total time of QPU access is {total_qpu_access_time*1e-6} seconds")
        print(f"Post-processing used {total_prostprocess_time} seconds")

    
