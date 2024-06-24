import yaml
from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learningv2
import sys
import os
import copy
# Node:   50    100     150     200
# Center: 37    57      70      75
NB_ACTIONS = 37
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def log(net, mcs, q_learning):
    while True:
        yield net.env.timeout(100)
        print_state_net(net, mcs)
        #print(q_learning.list_request)


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
for i in range (0,5):
    networkIO = NetworkIO(r"C:/Users/Quoc Dat/Desktop/SOICT/KLTN/KLTN/physical_env/network/network_scenarios/hanoi1000n50.yaml")
    env, net = networkIO.makeNetwork()

    with open(r"C:/Users/Quoc Dat/Desktop/SOICT/KLTN/KLTN/physical_env/mc/mc_types/default.yaml",
            'r') as file:
        mc_argc = yaml.safe_load(file)
    mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc, nb_action=NB_ACTIONS) for _ in range(5)]
    #print(mc for mc in mcs)

    q_learning = Q_learningv2(net=net, nb_action=NB_ACTIONS, q_alpha=0.1+0.2*i, q_gamma=0.1)

    for id, mc in enumerate(mcs):   
        mc.env = env
        mc.net = net
        mc.id = id
        mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
        mc.state = q_learning.nb_action - 1
    print(q_learning.q_alpha,q_learning.q_gamma)
    net.mc_list = mcs
    x = env.process(net.operate(optimizer=q_learning))
    #env.process(log(net, mcs, q_learning))
    env.run(until=x)
