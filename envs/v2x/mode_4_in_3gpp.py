# mode_4_in_3gpp 2023/10/19 17:18
import math

import numpy as np

from .. import MultiAgentEnv
from exceptions import IllegalArgumentException
from .util import V2V_Calculator
from .util import V2I_Calculator
from .util import Channel


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.destinations = []


class Mode_4_in_3GPP(MultiAgentEnv):
    def __init__(
            self,
            map_shape="rectangle",
            grid_x_length=250,
            grid_y_length=433,
            lane_width=3.5,
            lane_amount=2,
            grid_x_amount=3,
            grid_y_amount=3,
            bs_position_x=375,
            bs_position_y=649.5,
            bs_ant_height=25,
            bs_ant_gain=8,
            bs_noise_figure=5,
            veh_amount=4,
            veh_ant_height=1.5,
            veh_ant_gain=3,
            veh_noise_figure=9,
            veh_velocity="absolute,36,m/s",
            carry_frequency=2,
            resource_blocks=4,
            bandwidth=1,
            V2V_decorrelation_distance=10,
            V2V_shadow_std=3,
            neighbor_amount=1,
            V2I_decorrelation_distance=50,
            V2I_shadow_std=8,
            payload_size=1060,
            time_budget=100,
            seed=7777,
    ):
        # map_args:
        self.map_shape = map_shape
        self.grid_x_length = grid_x_length
        self.grid_y_length = grid_y_length
        self.lane_width = lane_width
        self.lane_amount = lane_amount
        self.grid_x_amount = grid_x_amount
        self.grid_y_amount = grid_y_amount
        # bs_args:
        self.bs_position_x = bs_position_x
        self.bs_position_y = bs_position_y
        self.bs_ant_height = bs_ant_height
        self.bs_ant_gain = bs_ant_gain
        self.bs_noise_figure = bs_noise_figure
        # veh_args:
        self.veh_amount = veh_amount
        self.veh_ant_height = veh_ant_height
        self.veh_ant_gain = veh_ant_gain
        self.veh_noise_figure = veh_noise_figure
        self.veh_velocity = veh_velocity
        # V2X_args
        self.carry_frequency = carry_frequency
        self.resource_blocks = resource_blocks
        self.bandwidth = bandwidth
        #  V2V_args
        self.V2V_decorrelation_distance = V2V_decorrelation_distance
        self.V2V_shadow_std = V2V_shadow_std
        self.neighbor_amount = neighbor_amount
        #  V2I_args
        self.V2I_decorrelation_distance = V2I_decorrelation_distance
        self.V2I_shadow_std = V2I_shadow_std
        self.payload_size = payload_size
        self.time_budget = time_budget
        self.seed = seed

        # compute useful args
        self.map_x_length = self.grid_x_length * self.grid_x_amount
        self.map_y_length = self.grid_y_length * self.grid_y_amount
        self.time_slow = self.time_budget

        # utils
        self.channel = None
        self._init_map()

    def reset_game(self):
        self._init_vehicles()
        self._reset_payload()
        self._reset_time()
        self.channel = Channel(self.vehicles,
                               self.delta_distance,
                               self.veh_amount,
                               self.neighbor_amount,
                               self.resource_blocks,
                               self.carry_frequency,
                               self.veh_ant_height,
                               self.V2V_decorrelation_distance,
                               self.V2V_shadow_std,
                               self.bs_ant_height,
                               self.V2I_decorrelation_distance,
                               self.V2I_shadow_std,
                               self.bs_position_x,
                               self.bs_position_y
                               )

    def _init_map(self):
        self.lanes = {
            'u': [i * self.grid_x_length + self.lane_width / 2 + self.lane_width * j
                  for i in range(self.grid_x_amount)
                  for j in range(self.lane_amount)],
            'd': [(i + 1) * self.grid_x_length - self.lane_width / 2 - self.lane_width * j
                  for i in range(self.grid_x_amount)
                  for j in range(self.lane_amount - 1, -1, -1)],
            'l': [i * self.grid_y_length + self.lane_width / 2 + self.lane_width * j
                  for i in range(self.grid_y_amount)
                  for j in range(self.lane_amount)],
            'r': [(i + 1) * self.grid_y_length - self.lane_width / 2 - self.lane_width * j
                  for i in range(self.grid_y_amount)
                  for j in range(self.lane_amount - 1, -1, -1)]
        }
        self.up_lanes = self.lanes['u']
        self.down_lanes = self.lanes['d']
        self.left_lanes = self.lanes['l']
        self.right_lanes = self.lanes['r']

        self.directions = list(self.lanes.keys())

    def _init_vehicles(self):
        self.vehicles = []

        def add_vehicle():
            direction = np.random.choice(self.directions)
            road = np.random.randint(0, len(self.lanes[direction]))
            if direction == 'u' or direction == 'd':
                x = self.lanes[direction][road]
                y = np.random.rand() * self.map_y_length
            else:
                x = np.random.rand() * self.map_x_length
                y = self.lanes[direction][road]
            position = [x, y]
            self.vehicles.append(Vehicle(position, direction, get_velocity()))

        def get_velocity():
            opts = self.veh_velocity.split(',')
            try:
                x = opts[0].split('-')
                if len(x) == 1:
                    v = float(x[0])
                    if opts[1].lower() == 'km/s':
                        v /= 3.6
                    return v
                else:
                    low, high = (float(x[0]), float(x[1]))
                    if opts[1].lower() == 'km/s':
                        low /= 3.6
                        high /= 3.6
                    return low + np.random.rand() * (high - low)

            except Exception:
                raise IllegalArgumentException('veh_velocity', self.veh_velocity,
                                               "Expected {>0},{km/s or m/s}")

        for i in range(self.veh_amount):
            add_vehicle()
        self.delta_distance = np.array([c.velocity * self.time_slow for c in self.vehicles])

    def _reset_payload(self):
        self.remain_payload = np.full((self.veh_amount, self.neighbor_amount), self.payload_size)

    def _reset_time(self):
        self.remain_time = self.time_budget

    def _renew_positions(self):
        # 论文作者给出的位置更新函数，基于该函数可以得到与论文相对接近的结果
        i = 0
        while i < len(self.vehicles):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if not change_direction:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if not change_direction:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.map_x_length) or (
                    self.vehicles[i].position[1] > self.map_y_length):
                # delete
                #    print ('delete ', self.position[i])
                if self.vehicles[i].direction == 'u':
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if self.vehicles[i].direction == 'd':
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if self.vehicles[i].direction == 'l':
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if self.vehicles[i].direction == 'r':
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1


    def step(self, actions):
        pass

    def get_obs(self):
        pass

    def get_obs_agent(self, agent_id):
        pass

    def get_obs_size(self):
        pass

    def get_state(self):
        pass

    def get_state_size(self):
        pass

    def get_avail_actions(self):
        pass

    def get_avail_agent_actions(self, agent_id):
        pass

    def get_total_actions(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
