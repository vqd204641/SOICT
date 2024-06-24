import yaml
from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learningv2
import sys
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Node:   50    100     150     200
# Center: 37    57      70      75
NB_ACTIONS = 37

def draw_network(nodes, base_station, targets,mobile_chargers, charging_positions):
    # Extract x, y coordinates and energy levels for plotting
    node_x = [node.location[0] for node in nodes]
    node_y = [node.location[1] for node in nodes]
    #node_energy = [node.energy for node in nodes]
    
    mc_x = [mc.current[0] for mc in mobile_chargers]
    mc_y = [mc.current[1] for mc in mobile_chargers]
    #mc_energy_percent = [mc.energy / mc.capacity * 100 for mc in mobile_chargers]

    base_station_x = base_station.location[0]
    base_station_y = base_station.location[1]

    target_x = [target.location[0] for target in targets]
    target_y = [target.location[1] for target in targets]

    #charging_pos_x = [charging_positions[i][0] for i in range(len(charging_positions))]
    #charging_pos_y = [charging_positions[i][1] for i in range(len(charging_positions))]

    # Plotting
    #plt.figure(figsize=(8, 8))  # Set figure size
    #fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(node_x, node_y, color='blue', label='Nodes') # Nodes in blue
    plt.scatter(mc_x, mc_y, color='black',marker= "h",s=50, label='MCs') # mc in black
    plt.scatter(target_x, target_y, color='red', marker='p', s=10, label='Targets')  # Targets in red squares
    plt.scatter(base_station_x, base_station_y, color='green', marker='s', s=50, label='Base Station')  # Base station in green triangle
    #plt.scatter(charging_pos_x, charging_pos_y, color='purple', marker='v', s=50, label='Charging position')  # Base station in green triangle

    for node in nodes:
        plt.text(node.location[0], node.location[1] - 25, f'{int(node.energy / node.capacity * 100)}%', fontsize=8, ha='center')

    for mc in mobile_chargers:
        plt.text(mc.current[0], mc.current[1] + 25, f'MC {mc.id}',color='red', fontsize=8, ha='center')

    # for pos in range(len(charging_positions)):
    #     plt.text(charging_positions[pos][0], charging_positions[pos][1] - 35, f'pos{pos}',color='purple', fontsize=8, ha='center')

    plt.title('Network Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    plt.xlim(0, 1000)  # Set x-axis limits
    plt.ylim(0, 1000)  # Set y-axis limits

    plt.draw()
    plt.pause(1) 
    #plt.grid(True)
    # plt.show()

def log(net, mcs, q_learning):
    while True:
        yield net.env.timeout(100)
        print_state_net(net, mcs)
        draw_network(net.listNodes, net.baseStation, net.listTargets,net.mc_list,net.network_cluster)
        plt.clf()

def print_state_net(net, mcs):
    print("[Network] Simulating time: {}s, lowest energy node is: id={} energy={:.2f} at {}".format(
        net.env.now, net.min_node(), net.listNodes[net.min_node()].energy, net.listNodes[net.min_node()].location))
    for mc in net.mc_list:
        if mc.chargingTime != 0 and mc.get_status() == "charging":
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy,
                                                                                         mc.get_status(),
                                                                                         mc.current,
                                                                                         mc.state))
        elif mc.moving_time != 0 and mc.get_status() == "moving":
            print("\t\tMC #{} energy:{} is {} to {} state:{}".format(mc.id, mc.energy,
                                                                                            mc.get_status(), mc.end,
                                                                                            mc.state))
        else:
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy, mc.get_status(), mc.current,
                                                                     mc.state))

networkIO = NetworkIO(r"C:/Users/Quoc Dat/Desktop/SOICT/KLTN/KLTN/physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()

with open(r"C:/Users/Quoc Dat/Desktop/SOICT/KLTN/KLTN/physical_env/mc/mc_types/default.yaml",
          'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc,nb_action=NB_ACTIONS) for _ in range(5)]
#print(mc for mc in mcs)
q_learning = Q_learningv2(net=net, nb_action=NB_ACTIONS, q_alpha=0.7, q_gamma=0.1)

for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id  
    mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
    mc.state = q_learning.nb_action - 1
print("start program")
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
env.process(log(net, mcs, q_learning))
env.run(until=x)
