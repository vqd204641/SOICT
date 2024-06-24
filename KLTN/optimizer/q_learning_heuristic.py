import random

import numpy as np
from scipy.spatial import distance

from optimizer.utils import init_function, q_max_function, reward_function
from physical_env.network import Node


class Q_learningv2:
    def __init__(self, init_func=init_function, nb_action=37, theta=0.3, q_alpha=0.5, q_gamma=0.01,
                 load_checkpoint=False, net=None):
        self.action_list = np.zeros(nb_action + 1)
        self.nb_action = nb_action + 1
        self.q_table = init_func(nb_action=nb_action + 1)
        self.q_table_A = init_func(nb_action = nb_action + 1)
        self.q_table_B = init_func(nb_action = nb_action + 1)
        # self.state = nb_action+1
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        #self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        #self.theta = theta # e_save = node_threshold + theta * node_max_e
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.update_Q_A = False
        self.eps_double_q = 0.5

    def update_double_Q(self, mc, network, time_stem, q_max_func=q_max_function):
        if not self.list_request:
            return self.action_list[mc.state], 0

        self.update_Q_A =random.random() > 0.5
        
        mc_state_now = mc.state
        is_random = self.choose_next_state_double_Q(mc, network)
        self.set_reward(mc=mc, time_stem=time_stem, network=network)

        if self.update_Q_A:
            max_q_value = self.q_max_double_Q(mc, q_max_func, q_table=mc.q_table_B)
            mc.q_table_A[mc_state_now] = (1-self.q_alpha)*mc.q_table_A[mc_state_now] + self.q_alpha * (self.reward + self.q_gamma * max_q_value)
        else:
            max_q_value = self.q_max_double_Q(mc, q_max_func, q_table=mc.q_table_A)
            mc.q_table_B[mc_state_now] = (1-self.q_alpha)*mc.q_table_B[mc_state_now] + self.q_alpha * (self.reward + self.q_gamma * max_q_value)
        
        # if self.update_Q_A:
        #     max_q_value = mc.q_table_B[mc.state][np.argmax(mc.q_table_A[mc.state])]
        #     mc.q_table_A[mc_state_now][mc.state] = (1-self.q_alpha)*mc.q_table_A[mc_state_now][mc.state] + self.q_alpha * (self.reward[mc.state] + self.q_gamma * max_q_value)
        # else:
        #     max_q_value = mc.q_table_A[mc.state][np.argmax(mc.q_table_B[mc.state])]
        #     mc.q_table_B[mc_state_now][mc.state] = (1-self.q_alpha)*mc.q_table_B[mc_state_now][mc.state] + self.q_alpha * (self.reward[mc.state] + self.q_gamma * max_q_value)

        if mc.state == len(self.action_list) - 1:
            charging_time = 0   

        if mc.state == len(self.action_list) - 1:
            charging_time = 0        
        # elif is_random:
        #     charging_time = 15
        else:
            charging_time = self.charging_time[mc.state]
        return self.action_list[mc.state], charging_time


    # def update(self, mc, network, time_stem, q_max_func=q_max_function):
    #     if not self.list_request:
    #         return self.action_list[mc.state], 0
    #     mc_state_now = mc.state
        
    #     self.choose_next_state(mc, network)
    #     self.set_reward(mc=mc, time_stem=time_stem, network=network)

    #     self.q_table[mc_state_now] = (1 - self.q_alpha) * self.q_table[mc_state_now] + self.q_alpha * (
    #             self.reward + self.q_gamma * self.q_max(mc, q_max_func))
    #     if mc.state == len(self.action_list) - 1:
    #         charging_time = 0
    #     else:
    #         charging_time = self.charging_time[mc.state]
    #     return self.action_list[mc.state], charging_time

    def q_max(self, mc, q_max_func=q_max_function):
        return q_max_function(q_table=self.q_table, state=mc.state)

    def q_max_double_Q(self, mc, q_max_func=q_max_function, q_table = None):
        return q_max_function(q_table=q_table, state=mc.state)

    def set_reward(self, mc=None, time_stem=0, network=None):
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_function(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third

        # first = first[mc.state] / np.sum(first)
        # second = second[mc.state] / np.sum(second)
        # third = third[mc.state] / np.sum(third)
        # self.reward[mc.state] = first + second + third
        # self.reward = first + third
       
        #self.reward_max = list(zip(first, second, third))

    # def choose_next_state(self, mc, network):
    #     # next_state = np.argmax(self.q_table[mc.state])
    #     if mc.energy < mc.threshold:  # 10
    #         mc.state = len(self.q_table) - 1
    #         print('[Optimizer] MC #{} energy is running low ({:.2f}), and needs to rest!'.format(mc.id, mc.energy))
    #     elif random.random() > self.eps_double_q:
    #         mc.state = random.randint(0, self.nb_action - 2)
    #         self.eps_double_q += 0.01
    #     else:
    #         mc.state = np.argmax(self.q_table[mc.state])
    #         if mc.state == len(self.q_table) - 1:
    #             mc.state = random.randrange(len(self.q_table)-1)

    def choose_next_state_double_Q(self, mc, network):
        # next_state = np.argmax(self.q_table[mc.state])
        is_random = False
        if mc.energy < mc.threshold:  # 10
            mc.state = len(self.q_table) - 1
            # print(mc.state)
        elif random.random() > mc.eps_double_q:
            mc.state = random.randint(0, self.nb_action - 2)
            mc.eps_double_q += 0.01
            is_random = True
        else:
            # q_table_A_B_state = mc.q_table_A[mc.state] + mc.q_table_B[mc.state]
            # q_table_A_B_state[mc.state] = -float("inf")
            mc.state = np.argmax(mc.q_table_A[mc.state] + mc.q_table_B[mc.state])
            while mc.state >= len(self.q_table) - 1:
                mc.state = random.randrange(len(self.q_table) - 2)
        
        return is_random