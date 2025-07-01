from crowd_nav.policy.policy import Policy
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY

'''
The base class for all policies
'''
class SRNN(Policy):
	def __init__(self, config):
		super().__init__(config)
		self.time_step = self.config.env.time_step # Todo: is this needed?
		self.name = 'srnn'
		self.trainable = True
		self.multiagent_training = True


	# clip the self.raw_action and return the clipped action
	def clip_action(self, raw_action, v_pref):
		"""
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
		# quantize the action
		holonomic = True if self.config.action_space.kinematics == 'holonomic' else False
		# clip the action
		if holonomic:
			act_norm = np.linalg.norm(raw_action)
			if act_norm > v_pref:
				raw_action[0] = raw_action[0] / act_norm * v_pref
				raw_action[1] = raw_action[1] / act_norm * v_pref
			return ActionXY(raw_action[0], raw_action[1])
		else:
			# for sim2real
			raw_action[0] = np.clip(raw_action[0], -0.1, 0.1) # action[0] is change of v
			# raw[0, 1] = np.clip(raw[0, 1], -0.25, 0.25) # action[1] is change of w
			# raw[0, 0] = np.clip(raw[0, 0], -state.self_state.v_pref, state.self_state.v_pref) # action[0] is v
			raw_action[1] = np.clip(raw_action[1], -0.1, 0.1) # action[1] is change of theta

			return ActionRot(raw_action[0], raw_action[1])

class selfAttn_merge_SRNN(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'selfAttn_merge_srnn'

class selfAttn_merge_SRNN_lidar(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'selfAttn_merge_SRNN_lidar'

class DSRNN_obs_vertex(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'dsrnn_obs_vertex'

class Homo_Transformer_Obs(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'homo_transformer_obs'

class lidar_gru(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'lidar_gru'
