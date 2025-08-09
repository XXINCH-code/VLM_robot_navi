import numpy as np

'''
Configuration Sections:
- Environment (`env`): Defines parameters for the environment type, scenario, and time limits.
- Robot (`robot`): Sets up the robot's physical and behavioral characteristics, such as visibility and policy.
- Human (`humans`): Configures human behavior, visibility, and policies.
- Reward (`reward`): Establishes the reward structure, including success, collision, and discomfort penalties.
- Sensors (`lidar`, `camera`): Specifies configurations for the robot's lidar and camera systems.
- Training (`training`): Contains parameters for training setups, including PPO hyperparameters, logging, and resumption options.
- Planning (`planner`): Provides configuration for the A* planner and waypoint sampling, particularly for navigation paths.
'''


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # environment settings
    env = BaseConfig()
    # all other policies: 'CrowdSim3DTbObs-v0'
    # A*+CNN: 'CrowdSim3DTbObsHieTrain-v0'
    env.env_name = 'CrowdSim3DTbObs-v0'  # name of the gym environment
    env.action_space = 'discrete'  # discrete or continuous action space
    # recommended value: if goal dist in [7, 9]: 30, if goal dist < 5: 20
    env.time_limit = 50  # time limit of each episode (second)
    env.time_step = 0.1  # length of each timestep/control frequency (second)
    env.val_size = 100
    env.test_size = 500  # number of episodes for test.py
    env.randomize_attributes = False  # randomize the preferred velocity and radius of humans or not
    env.seed = 50569  # random seed for environment
    # circle_crossing: circle crossing humans, random robot init & goal poses, random obstacles
    # csl_workspace: human flow in a set of regions, robot init & goal poses in a set of regions, fixed obstacles
    env.scenario = 'csl_workspace'
    # sim or sim2real
    env.mode = 'sim2real'

    # whether to use VLM for navigation, if "real", the robot will detect env and human activity by vlm
    env.use_vlm = True  # False means sim; True means real
    env.use_activity_weight = True  # whether to use human activity weight in attention mechanism
    env.random_env = True  # whether to use random environment, if False, use fixed environment
    env.test_in_pybullet = False  # whether to test the model with VLM in pybullet or not, if True, use VLM to  analyse images
    env.human_activity_beta = 2  # beta value for human activity weight, used in attention mechanism
    env.hard_block_th = 1.0       # 低于该权重的人的注意力在 softmax 前直接屏蔽
    env.activity_gain_cap = 1.8
    env.carrying_v_cap = 0.3
    env.talking_wall_width = 0.5

    # robot action type
    action_space = BaseConfig()
    # holonomic or unicycle or turtlebot
    action_space.kinematics = "turtlebot"

    ob_space = BaseConfig()
    # the robot state contains absolute positions [px, py, gx, gy] or relative positions [gx-px, gy-py]
    # note: for best result, relative positions require info on static obstacles
    ob_space.robot_state = 'absolute'  # absolute or relative
    # True: human observation is [px, py, vx, vy], False: human observation is [px, py]
    if env.mode == 'sim':
        ob_space.add_human_vel = True
    else:
        ob_space.add_human_vel = False
    # include humans + obs in lidar pc, or only include obs
    # todo: change this
    ob_space.lidar_pc_include_humans = True
    # the human states are in robot frame or world frame
    if env.mode == 'sim':
        ob_space.human_state_frame = 'robot'
    else:
        ob_space.human_state_frame = 'world'
    # the human velocity values are absolute (w.r.t. a static frame) or relative (w.r.t. the robot's velocity)
    ob_space.human_vel = 'absolute'

    # reward function
    reward = BaseConfig()
    reward.success_reward = 20
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.3
    reward.discomfort_penalty_factor = 15
    # dynamic navigation reward (diff envs)
    reward.keep_right_coeff = 0.1 # original 0.3
    # penalty for robot speed in corner
    reward.corner_speed_penalty = 0.2  # original 0.2
    # reduce the potential reward for hierarchical policy with A*
    if 'Hie' in env.env_name:
        reward.potential_reward_factor = 1
    else:
        reward.potential_reward_factor = 2
    if action_space.kinematics == 'unicycle':
        reward.spin_factor = 4.5
        reward.back_factor = 0.5
    elif action_space.kinematics == 'turtlebot':
        reward.spin_factor = 0.05
        reward.back_factor = 0.
    else:
        reward.spin_factor = 0
        reward.back_factor = 0
    # a constant penalty subtracted at every timestep, to prevent robot timeout especially when the task horizon is long
    reward.constant_penalty = -0.025
    # for hierarchical policy only
    reward.waypoint_reward = 1
    reward.gamma = 0.99  # discount factor for rewards

    # environment settings
    sim = BaseConfig()
    # controls the agent positions
    sim.circle_radius = 4
    # sim.robot_circle_radius = 5
    sim.robot_circle_radius = 4
    # number of dynamic humans
    sim.human_num = 7
    # the range of human_num is human_num-human_num_range~human_num+human_num_range
    sim.human_num_range = 2
    # number of static humans
    sim.static_human_num = 1
    sim.static_human_range = 1
    # actual human num is in [human_num-human_num_range, human_num+human_num_range]
    # warning: may have problems if human_num - human_num_range < observed_human_num

    # change human num within an episode periodically
    sim.change_human_num_in_episode = False
    # Group environment: set to true; FoV environment: false
    sim.group_human = False
    sim.human_pos_noise_range = 2
    # add static obstacles or not
    sim.static_obs = True
    # the position and size of obstacles are random or fixed
    if env.scenario == 'circle_crossing':
        sim.random_obs = True
        sim.obs_size_mean = 1
        sim.obs_size_std = 0.6
        sim.obs_max_size = 5
        sim.obs_min_size = 0.1
    else:
        sim.random_obs = False
    sim.static_obs_num = 10
    sim.static_obs_num_range = 2
    # whether we allow obstacles to overlap
    sim.obs_can_overlap = False
    # minimal distance between each pair of obstacles
    sim.obs_min_dist = 1
    # randomize the height of obstacles or not (if True, some obs will be too short and not detectable by lidar)
    sim.random_static_obs_height = False
    # add borders or not, the border will be a square centered at (0, 0) with width = 2*sim.arena_size
    sim.borders = True
    if env.scenario == 'csl_workspace':
        sim.borders = False # to get figures in the paper (without checkerboard floor), set to True during testing

    # render the simulation during training or not
    sim.render = False



    # robot settings
    robot = BaseConfig()
    robot.visible = True  # the robot is visible to humans
    # If robot.visible = true, the probability that a human will react to the robot
    robot.visible_prob = 0.2
    # robot policy, with only human positions: selfAttn_merge_srnn (Liu et al, ICRA 2023)
    # robot policy, with only obstacle positions: dsrnn_obs_vertex (Liu et al, ICRA 2021)
    # A*+CNN: lidar_gru (Perez-D’Arpino et al)
    # (ours & ablation) with both human positions and lidar: selfAttn_merge_srnn_lidar
    # (homogeneous attention graph network) homo_transformer_obs
    robot.policy = 'selfAttn_merge_srnn_lidar'

    if action_space.kinematics == "turtlebot":
        robot.radius = 0.15
    else:
        robot.radius = 0.3  # radius of the robot
    robot.height = 0.45  # height of the robot  
    robot.v_pref = 1  # max velocity of the robot
    robot.allow_backward = True
    # for turtlebot
    robot.v_max = 0.5
    if not robot.allow_backward:
        robot.v_min = 0
        reward.back_factor = 0.
    else:
        robot.v_min = -0.5
        reward.back_factor = 0.1
    robot.w_max = 1.5
    robot.w_min = -1.5
    # robot FOV = this values * PI
    robot.FOV = 2.
    # include (gx, gy) in the robot state in observation or not
    robot.visual_goal = True

    # for both circle_crossing and csl_workspace
    # range of distance between robot initial position and goal position
    # if you don't want to specify the range, set robot.min_goal_dist = 0 and robot.max_goal_dist = np.inf
    robot.min_goal_dist = 5  # 2
    robot.max_goal_dist = 6 # 4
    if env.mode == 'sim':
        robot.initTheta_range = [0, 2 * np.pi]
    else:
        robot.initTheta_range = [np.pi/2 - np.pi/6, np.pi/2 + np.pi/6]
    # for circle_crossing only
    # range of robot initial positions
    robot.initX_range = [-sim.robot_circle_radius, sim.robot_circle_radius]
    robot.initY_range = [-sim.robot_circle_radius, sim.robot_circle_radius]

    # range of robot goal positions
    robot.goalX_range = [-sim.robot_circle_radius, sim.robot_circle_radius]  # [-1.5, 0.4]
    robot.goalY_range = [-sim.robot_circle_radius, sim.robot_circle_radius]  # [7, 9]


    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = False
    # test ROS navigation stack or ours
    sim2real.test_nav_stack = False
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True
    # zed: only use zed2 camera to detect people
    # lidar: only use DR_SPAAM + LiDAR to detect people
    # fusion: use zed2 for people > 1m w.r.t. robot, use lidar for people < 1m w.r.t. robot
    sim2real.human_detector = 'lidar'
    sim2real.robot_localization = 't265'

    # LIDAR config
    lidar = BaseConfig()
    lidar.add_lidar = True
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 2  # todo: 1
    # lidar range: see robot.sensor_range
    # the height of the lidar mounting point from floor
    lidar.height = 0.5
    lidar.sensor_range = 25  # based on official document of RPLidar R3
    lidar.visualize_rays = False  # should always be false to speed up training and testing without GUI

    # camera config
    camera = BaseConfig()
    # camera field of view (in degrees)
    camera.fov = robot.FOV * 90
    # angular resolution (offset angle between neighboring rays) in degrees
    camera.ray_angular_res = 2
    # mounting height of the camera
    camera.height = 0.5
    # width and height of the camera image in pixels
    camera.render_cam_fov = 120
    camera.render_cam_img_width = 900 # * 2
    camera.render_cam_img_height = 900 # * 2
    camera.render_checkpoint = None # should always be None, will be changed in test.py

    # human settings
    humans = BaseConfig()
    humans.visible = True  # a human is visible to other humans and the robot
    # policy to control the humans: orca or social_force
    humans.policy = "orca"
    humans.radius = 0.2 # radius of each human # original 0.25
    humans.height = 0.7  # height of each human
    humans.v_pref = 2  # max velocity of each human
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = True

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3
    humans.random_policy_changing = False

    # a human may have diffrferent activities
    humans.dynamic_activity = ['walking', 'carrying']
    humans.static_activity = ['static', 'talking']

    # add noise to observation or not
    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.1
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for dwa
    dwa = BaseConfig()
    dwa.predict_time = 0.5
    dwa.to_goal_cost_gain = 0.1
    dwa.speed_cost_gain = 0.8
    dwa.obstacle_cost_gain = 1.0
    dwa.robot_stuck_flag_cons = 0.008
    dwa.dynamics_weight = 4.0
    dwa.stuck_action = 2

    # how much does a point move in the velocity space
    dwa.v_resolution = 0.05
    dwa.yaw_rate_resolution = 0.1

    # These two values are used to calculate action. They only need to be changed if action changes
    # max_accel * dt = dv and max_delta_yaw_rate * dt = dw
    dwa.max_accel = 0.5
    dwa.max_delta_yaw_rate = 1.0

    # 0 refers to circle and 1 refers to rectangle
    dwa.robot_type = 0

    # if robot is rectagular robot, then it needs robot width and length
    dwa.robot_width = 0.2
    dwa.robot_length = 0.2

    # left bottom coordinate of the boundary
    dwa.boundary = np.array([-6, -6])
    dwa.boundary_width = 12
    dwa.boundary_height = 12

    # default obstacle
    # Can be anything. The obstacle will be updated as soon as program starts
    dwa.ob = np.array([[-1, -1],
                        [0, 2],
                        [4.0, 2.0],
                        [5.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 6.0],
                        [5.0, 9.0],
                        [8.0, 9.0],
                        [7.0, 9.0],
                        [8.0, 10.0],
                        [9.0, 11.0],
                        [12.0, 13.0],
                        [12.0, 12.0],
                        [15.0, 15.0],
                        [13.0, 13.0]
                    ])


    # cofig for RL ppo
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.01  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter

    # network config
    SRNN = BaseConfig()
    SRNN.robot_embedding_size = 64
    SRNN.obs_embedding_size = 64
    SRNN.human_embedding_size = 64
    # RNN size
    SRNN.human_node_rnn_size = 128 # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 128 # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    # Attention vector dimension
    SRNN.hr_attention_size = 128  # robot-human Attention size
    SRNN.ho_attention_size = 128  # obstacle-human Attention size

    # for self attention
    SRNN.use_hr_attn = True  # RH attn
    SRNN.hr_attn_head_num = 1  # number of attention heads for RH attn
    SRNN.use_self_attn = True  # HH attn
    SRNN.self_attn_size = 128


    # training config
    training = BaseConfig()
    training.lr = 5e-5 # 1e-4  # learning rate (default: 8e-5)
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    training.num_env_steps = 200e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    training.use_linear_lr_decay = True  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    training.use_proper_time_limits = False  # compute returns taking into account time limits
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.cuda = True  # use CUDA for training
    training.num_processes = 28  # was 16, how many training CPU processes to use
    # the saving directory for train.py
    #training.output_dir = 'data/ours_RH_HH_cornerEnv_with_staticHuman' 
    #training.output_dir = 'data/ours_RH_HH_corridorEnv' 
    training.output_dir = 'data/ours_RH_HH_cornerEnv' 
    # resume training from an existing checkpoint or not
    # none: train RL from scratch, rl: load a RL weight
    training.resume = 'rl'
    # if resume != 'none', load from the following checkpoint
    #training.load_path = 'trained_models/ours_HH_RH_randEnv/checkpoints/237800.pt'
    training.load_path = 'data/ours_RH_HH_cornerEnv/checkpoints/51000.pt'
    #training.load_path = 'data/ours_RH_HH_cornerEnv_with_staticHuman/checkpoints/18000.pt'
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU


    # pybullet config
    # common env configuration
    pybullet = BaseConfig()
    pybullet.mediaPath = 'crowd_sim/pybullet/media/'  # os.path.join("Envs", "pybullet", "turtlebot", "media")  # objects' model
    # simulation frequency (Note: this is different from
    pybullet.sim_timestep = 1. / 240  # recommended by PyBullet official
    pybullet.frameSkip = int(env.time_step / pybullet.sim_timestep)  # TODO: choose 36 if the control method is rotPose


    planner = BaseConfig()
    # the size of a grid
    planner.grid_resolution = 0.25
    # the min distance between robot goal/init pos and any obs is robot.radius * 2
    if planner.grid_resolution >= robot.radius * 2:
        raise ValueError("Increase grid resolution to avoid robot init or goal position being occupied in self.om")
    # unit: number of grids, not meter!!!
    planner.path_clearance = 1
    # After A* generates a path, sample a waypoint every "planner.path_resolution" waypoints
    planner.num_waypoints = int(6)
    # the maximum distance between every 2 waypoints
    planner.max_waypoint_dist = 1.25
    # sample a waypoint at most every k waypoints from A*
    planner.max_waypoint_resolution = int(np.ceil(planner.max_waypoint_dist/planner.grid_resolution))
    # replan every n timesteps
    planner.replan = False
    planner.replan_freq = 30
    planner.om_inludes_human = False


    def __init__(self):
        self.fixed_obs = BaseConfig()
        self.human_flow = BaseConfig()
        self.update_workspace_config()  # Initialize all configuration settings based on the current workspace type

    def update_workspace_config(self):
        """Update configuration based on csl_workspace_type."""
        if self.env.random_env:
            # if use_vlm, randomly choose between corner and corridor
            # plan to add simple corner and simple corridor
            #self.env.csl_workspace_type = np.random.choice(['corner', 'corridor'])
            self.env.csl_workspace_type = np.random.choice(['simple_corner', 'simple_corridor'])
        else:
            self.env.csl_workspace_type = 'corner'
        
        # Update fixed_obs based on the new csl_workspace_type
        self.update_fixed_obs()
        self.update_arena_size()
        self.update_robot_regions()

    def update_arena_size(self):
        if self.env.mode == 'sim':
            self.sim.arena_size = 4.5
        else:
            # for om, om size = arena_size + 1
            if self.env.csl_workspace_type == 'corner':
                self.sim.arena_size = 10
            elif self.env.csl_workspace_type == 'corridor':
                self.sim.arena_size = 10
            elif self.env.csl_workspace_type == 'simple_corner':
                self.sim.arena_size = 10
            elif self.env.csl_workspace_type == 'simple_corridor':
                self.sim.arena_size = 10

    def update_robot_regions(self):
        if self.env.csl_workspace_type == 'corner':
            self.robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                            2: np.array([-0.3, 0.3, 5.5, 6]),
                            3: np.array([-4, -3, 10, 11]),
                            }
            # short-distance navigation
            self.robot.routes = [[1, 3]]
        elif self.env.csl_workspace_type == 'corridor':
            self.robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                            2: np.array([-0.3, 0.3, 5.5, 6]),
                            3: np.array([4, 5, 12.5, 13.5]),
                            }
            # short-distance navigation
            self.robot.routes = [[1, 3]]
        elif self.env.csl_workspace_type == 'simple_corner':
            self.robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                            2: np.array([-0.3, 0.3, 5.5, 6]),
                            3: np.array([-5, -4, 3.5, 4.5]),
                            }
            # short-distance navigation
            self.robot.routes = [[1, 3]]
        elif self.env.csl_workspace_type == 'simple_corridor':
            self.robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                            2: np.array([-0.3, 0.3, 5.5, 6]),
                            3: np.array([0, 1, 10, 11]),
                            }
            # short-distance navigation
            self.robot.routes = [[1, 3]]

    def update_fixed_obs(self):
        """Update the fixed_obs configuration based on csl_workspace_type."""
        # r, g, b, alpha
        # 人类的默认颜色设置为绿色（第二个）
        self.human_flow.colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1],
                            [1, 0, 0.5, 1], [0, 1, 0.5, 1], [0, 0.5, 1, 1], [0.5, 1, 0, 1], [0.5, 0, 1, 1],
                            [1, 0.5, 0.5, 1],[0.5, 1, 0.5, 1], [0.5, 0.5, 1, 1], [0.25, 0, 1, 1], [0, 1, 0.25, 1], [0.25, 0.25, 0, 1], [1, 0.25, 0, 1]]

        assert len(self.human_flow.colors) >= self.sim.human_num + self.sim.human_num_range

        # fix the obstacle and wall layout for the 2 sim2real environments
        if self.env.scenario == 'csl_workspace':
            # [width, height] of all obstacles
            # left vertical wall, right vertical wall,
            # 3 workstations (the middle two are combined) from upper to lower, the extra horizontal wall on the bottom (near 0, 0)
            # the upper left and upper right rooms, the horizontal wall on the upper right,
            # the vertical wall on the upper right, the vertical wall on the upper left

            self.fixed_obs.rectangle_height = 2  # height of rectangular obstacles

            if self.env.csl_workspace_type == 'corner':
                divider_width = 6
                self.fixed_obs.cylinder_radius = 0.5 # 0.45
                self.fixed_obs.cylinder_height = 0.7
                # only includes the first 3 lines of workstation from bottom
                # [width, height] of all obstacles
                self.fixed_obs.sizes = np.array([[10, 1300], [800, 10],
                                            [700, 360], [700, 360], [140, 250],
                                            [1100, 250], 
                                            # 两个桌子
                                            [450, 200], [450, 200],
                                            # vertical dividers that seperate desks and hallway
                                            [divider_width, 360], [divider_width, 360],
                                            [self.fixed_obs.cylinder_radius*200, self.fixed_obs.cylinder_radius*200],
                                            [self.fixed_obs.cylinder_radius*200, self.fixed_obs.cylinder_radius*200]
                                            ]) / 100.
                # [x, y] coordinates of lower left corners of all obstacles
                self.fixed_obs.positions_lower_left = np.array([[590, -150], [-200, -150],
                                                        [-800, 80], [-800, 600], [-240, -150],
                                                        [-500, 1140], 
                                                        # 两个桌子
                                                        [150, 750], [150, 250],
                                                        # vertical dividers that seperate desks and hallway
                                                        [-97-divider_width, 100], [-97-divider_width, 580],
                                                        [200 - self.fixed_obs.cylinder_radius*100, 600 - self.fixed_obs.cylinder_radius*100],
                                                        [200 - self.fixed_obs.cylinder_radius*100, 100 - self.fixed_obs.cylinder_radius*100]
                                                        ]) / 100.
                self.fixed_obs.shapes = np.array([1] * 10 + [0] * 2)

                # define human routes based on map
                self.human_flow.static_regions = np.array([#[320, 570, 400, 600], [320, 570, -100, 100],
                                                    [80, 130, 680, 900],
                                                    [-75, -25, 180, 400]
                                                    ]) / 100.
                # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
                # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
                self.human_flow.regions = {0: np.array([550, 570, -100, 0]) / 100.,
                                    1: np.array([400, 450, -150, -50]) / 100.,
                                    1.5: np.array([150, 300, -150, 0]) / 100.,
                                    2: np.array([0, 100, 100, 200]) / 100.,
                                    3: np.array([-50, 25, 425, 575]) / 100.,
                                    3.5: np.array([-50, 0, 700, 800]) / 100.,
                                    4: np.array([-50, 50, 1025, 1075]) / 100.,
                                    5: np.array([300, 500, 990, 1050]) / 100.,
                                    6: np.array([-400, -200, 1000, 1110]) / 100.,
                                    7: np.array([-650, -600, 470, 570]) / 100.,
                                    7.5: np.array([-550, -500, 470, 530]) / 100.,
                                    7.75: np.array([-750, -700, 470, 570]) / 100.,
                                    8: np.array([-750, -550, 1100, 1300]) / 100.,
                                    8.5: np.array([-600, -500, 1000, 1100]) / 100.,
                                    }
                # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
                self.human_flow.route_type = 'correlated'

                self.human_flow.routes = [
                                    # both human and robot's routes are straight lines
                                    [7, 7, 7, 7],
                                    [1, 2], [2, 3, 3.5],
                                    [6, 4.5, 5],
                                    # the human takes a turn and cross the robot
                                    [6, 4.5, 4],
                                    [7, 3.5, 3, 2],
                                    [7, 3.5, 4],
                                    # the human takes a turn and does not cross the robot
                                    [1, 2, 3, 3.5], [4, 4.5, 6],
                                    [3, 3.5, 7], [4, 4.5, 5],
                                    ]

                self.human_flow.correlated_routes = [
                    #corner
                    [[7, 7.5, 3, 2, 1], [8, 8, 8, 8, 8, 8, 8, 6, 4, 2, 1.5]],
                    [[7, 7.5, 3, 2, 1], [8, 8, 8, 8, 8, 8, 8, 6, 5]],
                    # 2 people
                    [[1.5, 4], [4, 2, 1.5]],
                    [[1.5, 3, 7],[6, 4, 3, 7.5]],
                    # 2 people corner
                    [[7, 7.5, 3, 2, 1], [7.5, 3, 2, 1], 
                    [8.5, 8.5, 8.5, 4, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 2]],
                    # many people
                    [[7, 3, 3.5, 4, 5], [7.5, 3, 3.5, 4, 5],
                    [7.75, 3, 2, 1, 0]]
                    
                ]

            elif self.env.csl_workspace_type == 'corridor':
                divider_width = 6
                self.fixed_obs.sizes = np.array([# walls
                                            [10, 1200], [400, 250],
                                            [10, 200], [10, 200], 
                                            [650, 10], [100, 10],
                                            [10, 1000], [10, 300], [250, 150], [150, 300],
                                            # vertical dividers that seperate desks and hallway
                                            [30, divider_width], [30, divider_width],
                                            [30, divider_width], [30, divider_width]
                                            ]) / 100.
                # [x, y] coordinates of lower left corners of all obstacles
                self.fixed_obs.positions_lower_left = np.array([# walls
                                                        [-100, 0], [-500, 0],
                                                        [-200, 1200], [740, 1200],
                                                        [90, 1200], [-200, 1200],
                                                        [90, -200], [90, 900], [-150, -300], [-500, -300],
                                                        # vertical dividers that seperate desks and hallway
                                                        [-100, 0], [100-30, 0],
                                                        [-350, -300], [-150-30, -300]                                                
                                                        ]) / 100.
                # 1: rectangular cube, 0: cylinder
                self.fixed_obs.shapes = np.array([1] * len(self.fixed_obs.sizes))

                # define human routes based on map

                self.human_flow.static_regions = np.array([[-175, -75, 1225, 1400], [30, 100, 750, 950],
                                                    ]) / 100.
                
                # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
                # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
                self.human_flow.regions = {0: np.array([400, 500, 1200, 1350]) / 100.,
                                    1: np.array([150, 300, 1250, 1350]) / 100.,
                                    1.5: np.array([-25, 25, 1150, 1230]) / 100.,
                                    2: np.array([-75, 0, 950, 1030]) / 100.,
                                    3: np.array([-0, 75, 650, 700]) / 100.,
                                    4: np.array([-75, 50, 400, 500]) / 100.,
                                    5: np.array([-0, 75, 200, 300]) / 100.,
                                    6: np.array([-75, 75, 0, 100]) / 100.,
                                    7: np.array([-300, -200, -130, -50]) / 100.,
                                    7.5: np.array([-300, -200, -300, -170]) / 100.
                                    }

                # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
                self.human_flow.route_type = 'correlated'

                self.human_flow.routes = [
                                    # human goes right
                                    [0, 1, 1.5, 2, 4, 6, 7],
                                    [2, 4, 6, 7, 8],
                                    # human goes left
                                    [0, 1, 1.5, 3, 5, 7],
                                    # human goes left and then right
                                    [1, 1.5, 3, 4, 6, 7],
                                    ]
                self.human_flow.correlated_routes = [
                    # human goes right,towards robot
                    [[1, 1.5, 2, 4, 6, 7, 7.5], [0, 1, 1.5, 2, 4, 6, 7, 7.5]],
                    # human goes left, then right
                    [[0, 1, 1.5, 2, 3, 4, 6, 7]],
                    [[1, 1.5, 3, 4, 6, 7], [0, 1, 1.5, 2, 4, 6, 7]],
                    # human walk torwards
                    [[0, 1, 1.5, 2, 4, 6, 7], [7, 6, 5, 3, 1.5]],
                ]

            elif self.env.csl_workspace_type == 'square':
                #fixed_obs.sizes = np.array([[10, 500]]) / 100.
                self.fixed_obs.sizes = np.array([[10, 500], [600, 10], [10, 300],
                                            [200, 300], [90, 400], [190, 100],
                                            ]) / 100.
                # [x, y] coordinates of lower left corners of all obstacles
                #fixed_obs.positions_lower_left = np.array([[-300, -100]]) / 100.
                self.fixed_obs.positions_lower_left = np.array([[-300, -100], [-300, 400], [290, 100],
                                                        [300, 100], [-300, -100], [-300, 300]
                                                        ]) / 100.
                self.fixed_obs.shapes = np.array([1] * 6)

                # define human routes based on map
                self.human_flow.static_regions = np.array([[310, 570, -100, 100], [310, 570, 350, 600],
                                                    [-80, 150, 150,300], [-80, 150, 500, 600]
                                                    ]) / 100.
                # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
                # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
                self.human_flow.regions = {1: np.array([300, 400, -220, -100]) / 100.,
                                    2: np.array([-80, 200, -250, -100]) / 100.,
                                    2.5: np.array([0, 100, 0, 100]) / 100.,
                                    3: np.array([-80, 125, 300, 500]) / 100.,
                                    4: np.array([-80, 125, 600, 800]) / 100.,
                                    4.5: np.array([0, 125, 875, 1000]) / 100.,
                                    5: np.array([300, 500, 890, 950]) / 100.,
                                    6: np.array([-400, -200, 900, 1010]) / 100.,
                                    7: np.array([-700, -600, 370, 470]) / 100.,
                                    7.5: np.array([-330, -200, 370, 430]) / 100.,
                                    }

                # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
                self.human_flow.route_type = 'correlated'

                self.human_flow.routes = [

                                    ]

                self.human_flow.correlated_routes = [

                ]

            elif self.env.csl_workspace_type == 'simple_corner':
                self.fixed_obs.sizes = np.array([
                                            [400,610], 
                                            [950, 250], [300, 800], 
                                            [10,200],
                                            ]) / 100.
                # [x, y] coordinates of lower left corners of all obstacles
                self.fixed_obs.positions_lower_left = np.array([
                                                        [-500, -250],
                                                        [-500, 540], [150, -250],
                                                        [-500, 350],
                                                        ]) / 100.
                self.fixed_obs.shapes = np.array([1] * 4)
                # define human routes based on map
                self.human_flow.static_regions = np.array([[100, 130, 150, 350],
                                                    [-70, 130, 500, 530],
                                                    ]) / 100.
                
                # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
                # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
                self.human_flow.regions = {0: np.array([-470, -400, 400, 500]) / 100.,
                                    1: np.array([-350, -300, 450, 530]) / 100.,
                                    1.5: np.array([-50, 50, 400, 500]) / 100.,
                                    2: np.array([-50, 100, -50, 50]) / 100.,
                                    3: np.array([100, 130, -300, -270]) / 100.,
                                    }
                # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
                self.human_flow.route_type = 'correlated'
                self.human_flow.routes = [
                                    [0, 1, 1.5, 2],
                                    [1, 1.5, 2],
                                    [3, 1.5, 1]
                                    ]
                self.human_flow.correlated_routes = [
                    # human goes right,towards robot
                    [[0, 0, 0, 0, 0, 0, 1, 2], [1, 2]],
                    # human goes left, then right
                    [[1, 2],[3, 1.5, 1]],
                    [[0, 0, 0, 0, 0, 1, 2]],
                    [[1,2]],
                ]

            # static area还没有设置，还有huamn flow
            elif self.env.csl_workspace_type == 'simple_corridor':
                # change the number of static humans
                #self.sim.static_human_num = 1
                #self.sim.static_human_range = 0
                self.fixed_obs.sizes = np.array([
                                            [300, 1500], [300, 1500], 
                                            [200, 10], [200, 10],
                                            ]) / 100.
                # [x, y] coordinates of lower left corners of all obstacles
                self.fixed_obs.positions_lower_left = np.array([
                                                        [-400, -300], [100, -300],
                                                        [-100, 1200], [-100, -300]
                                                        ]) / 100.
                self.fixed_obs.shapes = np.array([1] * 4)
                # define human routes based on map
                self.human_flow.static_regions = np.array([[-80, -50, 150, 350],
                                                           [50, 80, -350, -150],
                                                    ]) / 100.
                
                # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
                # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
                self.human_flow.regions = {
                                    0: np.array([-50, 50, 1150, 1200]) / 100.,
                                    1: np.array([-50, 50, 1050, 1100]) / 100.,
                                    2: np.array([-80, -20, 800, 900]) / 100.,
                                    2.5: np.array([20, 80, 800, 900]) / 100.,
                                    3: np.array([-80, -20, 550, 650]) / 100.,
                                    3.5: np.array([20, 80, 550, 650]) / 100.,
                                    4: np.array([-50, 25, -250, -300]) / 100.,
                                    }

                # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
                self.human_flow.route_type = 'correlated'

                self.human_flow.routes = [
                                    # human goes right
                                    [0, 2, 3, 4],
                                    [1, 2, 3, 4],
                                    # human goes left
                                    [0, 2.5, 3.5, 4],
                                    [1, 2.5, 3.5, 4],
                                    # human goes left and then right
                                    [0, 2.5, 3, 4],
                                    # human goes right and then left
                                    [1, 2, 3.5, 4],
                                    ]
                self.human_flow.correlated_routes = [
                    # human goes right,towards robot
                    [[0, 2, 3, 4], [1, 2, 3, 4]],
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4], [1, 2, 3, 4]],
                    # human goes left, then right
                    [[0, 2.5, 3.5, 4], [1, 2.5, 3.5, 4]],
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5, 3.5, 4], [1, 2.5, 3.5, 4]],
                    [[1, 2.5, 3.5, 4], [0, 2, 3, 4]],
                    [[1, 2.5, 3.5, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4]],

                    [[0, 2, 3, 4]],
                    [[0, 2.5, 3.5, 4]],
                    [[0, 2.5, 3, 4]],
                    [[0, 2, 3.5, 4]],
                ]
                
            else:
                raise ValueError("Unknown csl_workspace_type")
            assert len(self.fixed_obs.sizes) == len(self.fixed_obs.positions_lower_left)

            # change the static obs information based on the fixed_obs above
            self.sim.static_obs_num = len(self.fixed_obs.sizes)
            self.sim.static_obs_num_range = 0

            # make sure each route has a start region and at least one goal region
            for route in self.human_flow.routes:
                assert len(route) >= 2

            # adjust the human_num to prevent errors for correlated routes
            if self.human_flow.route_type == 'correlated':
                self.sim.human_num = self.sim.static_human_num + max(len(sublist) for sublist in self.human_flow.correlated_routes)
                self.sim.human_num_range = 0

        # for circle crossing sceanrio, humans start & goals are always sampled randomly
        else:
            self.human_flow.route_type = 'independent'

