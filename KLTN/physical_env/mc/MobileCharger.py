import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import copy
from optimizer.utils import init_function

# from torch.testing._internal.common_device_type import ops


class MobileCharger:
    def __init__(self, location, mc_phy_spe,nb_action,init_func=init_function):
        """
        The initialization for a MC.
        :param env: the time management system of this MC
        :param location: the initial coordinate of this MC, usually at the base station
        """
        self.chargingTime = 0
        self.env = None
        self.net = None
        self.id = None
        self.cur_phy_action = [500, 500, 0]
        self.location = np.array(location)
        self.energy = mc_phy_spe['capacity']
        self.capacity = mc_phy_spe['capacity']

        self.alpha = mc_phy_spe['alpha']
        self.beta = mc_phy_spe['beta']
        self.threshold = mc_phy_spe['threshold']
        self.velocity = mc_phy_spe['velocity']
        self.pm = mc_phy_spe['pm']
        self.chargingRate = 0
        self.chargingRange = mc_phy_spe['charging_range']
        self.epsilon = mc_phy_spe['epsilon']
        self.status = 1
        self.checkStatus()
        self.cur_action_type = "deactive"
        self.connected_nodes = []
        self.incentive = 0
        self.end = self.location
        self.start = self.location
        self.state = 37

        self.q_table_A = init_func(nb_action = nb_action + 1)
        self.q_table_B = init_func(nb_action = nb_action + 1)
        # self.q_table = []
        self.eps_double_q = 0.5
        self.next_phy_action = [500, 500, 0]
        self.save_state = []
        self.e = mc_phy_spe['e']
        self.is_active = False
        self.is_self_charge = False
        self.is_stand = False
        self.current = self.location
        self.end_time = 0
        self.moving_time = 0
        self.arrival_time = 0
        self.e_move = mc_phy_spe['velocity']
        self.next_location = [500, 500]

    def modify_alpha(self):
        if len(self.net.listTargets) > 50:
            self.alpha = 9000

    def charge_step(self, t):
        self.modify_alpha()
        """
        The charging process to nodes in 'nodes' within simulateTime
        :param nodes: the set of charging nodes
        :param t: the status of MC is updated every t(s)
        """
        for node in self.connected_nodes:
            node.charger_connection(self)

        # print("MC " + str(self.id) + " " + str(self.energy) + " Charging", self.location, self.energy, self.chargingRate)
        yield self.env.timeout(t)
        self.energy = self.energy - self.chargingRate * t
        self.cur_phy_action[2] = max(0, self.cur_phy_action[2] - t)
        for node in self.connected_nodes:
            node.charger_disconnection(self)
        self.chargingRate = 0
        return

    def chargev2(self, net):
        for nd in net.listNodes:
            p = nd.charge(mc=self)
            self.energy -= p

    def update_location(self):
        self.current = self.get_location()
        self.energy -= self.e_move

    def get_location(mc):
        d = distance.euclidean(mc.start, mc.end)
        time_move = d / mc.velocity
        if time_move == 0:
            return mc.current
        elif distance.euclidean(mc.current, mc.end) < 10 ** -3:
            return mc.end
        else:
            x_hat = (mc.end[0] - mc.start[0]) / time_move + mc.current[0]
            y_hat = (mc.end[1] - mc.start[1]) / time_move + mc.current[1]
            if (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) < 0 or (
                    (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) == 0 and (mc.end[1] - mc.current[1]) * (
                    mc.end[1] - y_hat) <= 0):
                return mc.end
            else:
                return x_hat, y_hat

    def move_step(self, vector, t):
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity

    def move(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        # self.end_time = moving_time
        self.arrival_time = moving_time
        moving_vector = destination - self.location
        total_moving_time = moving_time
        while True:
            if moving_time <= 0:
                break
            if self.status == 0:
                yield self.env.timeout(moving_time)
                break
            moving_time = euclidean(destination, self.location) / self.velocity
            # print("MC " + str(self.id) + " " + str(self.energy) + " Moving from", self.location, "to", destination)
            span = min(min(moving_time, 1.0), (self.energy - self.threshold) / (self.pm * self.velocity))
            # span = 1
            yield self.env.process(self.move_step(moving_vector / total_moving_time * span, t=span))
            moving_time -= span
            self.checkStatus()
        # print("energy after moving", self.energy)
        return self.arrival_time

    def move_time(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        self.arrival_time = moving_time
        return self.arrival_time

    def recharge(self):
        if euclidean(self.location, self.net.baseStation.location) <= self.epsilon:
            self.location = copy.deepcopy(self.net.baseStation.location)
            self.energy = self.capacity
        self.is_self_charge = True
        yield self.env.timeout(0)

    # def update_q_table(self, optimizer, net, time_stem):
    #     result = optimizer.update_v2(self, net, time_stem)
    #     self.q_table = result[1]
    #     self.next_phy_action = []
    #     self.next_phy_action = [result[2][0], result[2][1], result[3]]
    #     return result[0]

    def check_cur_action(self):
        if not self.cur_action_type == 'moving':
            self.cur_action_type = 'charging'
        elif not self.cur_action_type == 'charging':
            self.cur_action_type = 'recharging'
        elif not self.cur_action_type == 'recharging':
            self.cur_action_type = 'deactive'

    def get_status(self):
        if not self.is_active:
            return "deactivated"
        if not self.is_stand:
            return "moving"
        if not self.is_self_charge:
            return "charging"
        return "self_charging"

    def checkStatus(self):
        """
        check the status of MC
        """
        if self.energy <= self.threshold:
            # if self.energy <= 0:
            self.status = 0
            self.energy = self.threshold

    # def get_next_location(self, network, time_stem, optimizer=None):
    #     next_location, charging_time = optimizer.update(self, network, time_stem)
    #     self.start = self.current
    #     # self.cur_phy_action = [next_location[0], next_location[1], charging_time]
    #     self.end = next_location
    #     self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
    #     self.end_time = time_stem + self.moving_time + charging_time
    #     #print("[Moblie Charger] MC #{} end_time {}".format(self.id, self.end_time))
    #     self.chargingTime = charging_time
    #     self.arrival_time = time_stem + self.moving_time

    def get_next_locationv2(self, network, time_stem, optimizer=None):
        next_location, charging_time = optimizer.update_double_Q(self, network, time_stem)
        is_same_destination = False
        for other_mc in self.net.mc_list:
            if other_mc.id != self.id and euclidean(other_mc.end, next_location) < 0.1:
                is_same_destination = True
        if not is_same_destination:
            self.start = self.current
            # self.cur_phy_action = [next_location[0], next_location[1], charging_time]
            self.end = next_location
            self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
            self.end_time = time_stem + self.moving_time + charging_time
            #print("[Moblie Charger] different MC #{} end_time {}".format(self.id, self.end_time))
            self.chargingTime = charging_time
            #if self.chargingTime != 0:
                #print("[Moblie Charger] MC #{} charge in {}".format(self.id, self.chargingTime))
            self.arrival_time = time_stem + self.moving_time

    def runv2(self, network, time_stem, net=None, optimizer=None):
        if ((not self.is_active) and optimizer.list_request) or (np.abs(time_stem - self.end_time) < 10):
            self.is_active = True
            new_list_request = []
            for request in optimizer.list_request:
                if net.listNodes[request["id"]].energy < self.net.listNodes[0].warning:
                    new_list_request.append({"id": net.listNodes[request["id"]].id,
                                             "energy": net.listNodes[request["id"]].energy,
                                             "energyCS": net.listNodes[request["id"]].energyCS,
                                             "energyRR": net.listNodes[request["id"]].energyRR,
                                             "time": 1})
                else:
                    net.listNodes[request["id"]].is_request = False
            optimizer.list_request = new_list_request
            if not optimizer.list_request:
                self.is_active = False
            current_location = [self.end[0], self.end[1]]
            self.get_next_locationv2(network=network, time_stem=time_stem, optimizer=optimizer)
            base_loc = [500, 500]
            if euclidean(current_location, base_loc) < 0.1:
                self.is_active = False
                if len(optimizer.list_request) >= 3:
                    self.charge_lowest_e_sensor(self, net = net, time_stem=time_stem, optimizer=optimizer)

        else:
            if self.is_active:
                if not self.is_stand:
                    self.update_location()
                elif not self.is_self_charge:
                    self.chargev2(net)
                else:
                    self.recharge()
        if self.energy < self.threshold and not self.is_self_charge and self.end != self.net.baseStation.location:
            self.start = self.current
            self.end = self.net.baseStation.location
            self.is_stand = False
            charging_time = 0
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()

    # def __str__(self):
    #     return f"MobileCharger(id='{self.id}', location={self.location}, cur_action_type={self.cur_action_type})"

    def check_state(self):
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(self.net.baseStation.location, self.end) < 10 ** -3 and self.energy < 1500:
            self.is_self_charge = True
        else:
            self.is_self_charge = False

    def charge_lowest_e_sensor(self, network, time_stem, net=None, optimizer=None):
        min = float('inf')
        id_lowest_e_sensor = None
        for request in optimizer.list_request:
            if net.listNodes[request["id"]].energy < min:
                min = net.listNodes[request["id"]].energy
                id_lowest_e_sensor = request["id"]

        next_location = net.listNodes[id_lowest_e_sensor].location
        charging_time = 100
        self.start = self.current
        # self.cur_phy_action = [next_location[0], next_location[1], charging_time]
        self.end = next_location
        self.moving_time = distance.euclidean(self.location, self.end) / self.velocity
        self.chargingTime = charging_time
        #if self.chargingTime != 0:
           # print("[Moblie Charger] MC #{} charge in {}".format(self.id, self.chargingTime))
        self.end_time = time_stem + self.moving_time + charging_time
        #print("[Moblie Charger] auto-lowest-charge MC #{} end_time {}".format(self.id, self.end_time))
        self.arrival_time = time_stem + self.moving_time
        # self.is_active = True
