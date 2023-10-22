# v2i_calculator 2023/10/22 21:33
import math

import numpy as np


class V2I_Calculator:
    def __init__(self,
                 veh_ant_height,
                 bs_ant_height,
                 V2I_decorrelation_distance,
                 V2I_shadow_std,
                 bs_position
                 ):
        self.h_bs = bs_ant_height
        self.h_ms = veh_ant_height
        self.Decorrelation_distance = V2I_decorrelation_distance
        self.BS_position = bs_position  # center of the grids
        self.shadow_std = V2I_shadow_std

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) \
               * np.random.normal(0, self.shadow_std, len(shadowing))

