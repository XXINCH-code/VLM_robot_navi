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
    # if env.scenario == 'csl_workspace', the environment is 
    env.csl_workspace_type = 'hallway'  # hallway, lounge, corner, corridor
    # sim or sim2real
    env.mode = 'sim'

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
    ob_space.lidar_pc_include_humans = False
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
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
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
    # controls the obstacle positions
    if env.mode == 'sim':
        sim.arena_size = 4.5
    else:
        # for om, om size = arena_size + 1
        if env.csl_workspace_type == 'hallway':
            sim.arena_size = 6
        elif env.csl_workspace_type == 'lounge':
            sim.arena_size = 11
        elif env.csl_workspace_type == 'corner':
            sim.arena_size = 10
        elif env.csl_workspace_type == 'corridor':
            sim.arena_size = 10
    # number of dynamic humans
    sim.human_num = 7
    # the range of human_num is human_num-human_num_range~human_num+human_num_range
    sim.human_num_range = 2
    # number of static humans
    sim.static_human_num = 2
    sim.static_human_range = 0
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

    human_flow = BaseConfig()
    # r, g, b, alpha
    # 人类的默认颜色设置为绿色（第二个）
    human_flow.colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1],
                          [1, 0, 0.5, 1], [0, 1, 0.5, 1], [0, 0.5, 1, 1], [0.5, 1, 0, 1], [0.5, 0, 1, 1],
                         [1, 0.5, 0.5, 1],[0.5, 1, 0.5, 1], [0.5, 0.5, 1, 1], [0.25, 0, 1, 1], [0, 1, 0.25, 1], [0.25, 0.25, 0, 1], [1, 0.25, 0, 1]]
   
    assert len(human_flow.colors) >= sim.human_num + sim.human_num_range

    # fix the obstacle and wall layout for the 2 sim2real environments
    if env.scenario == 'csl_workspace':
        fixed_obs = BaseConfig()
        # [width, height] of all obstacles
        # left vertical wall, right vertical wall,
        # 3 workstations (the middle two are combined) from upper to lower, the extra horizontal wall on the bottom (near 0, 0)
        # the upper left and upper right rooms, the horizontal wall on the upper right,
        # the vertical wall on the upper right, the vertical wall on the upper left
        
        fixed_obs.rectangle_height = 2  # height of rectangular obstacles

        if env.csl_workspace_type == 'hallway':
            # define obstacles based on map
            divider_width = 6
            fixed_obs.cylinder_radius = 0.5 # 0.45
            fixed_obs.cylinder_height = 0.7
            # only includes the first 3 lines of workstation from bottom
            # [width, height] of all obstacles
            
            fixed_obs.sizes = np.array([[50, 808], [10, 992],
                                        [699, 84], [699, 84 * 2], [699, 84], [140, 250],
                                        [270 + 10, 250], [1137 + 10, 250], [650, 10],
                                        [10, 202], [10, 136],
                                        # vertical dividers that seperate desks and hallway
                                        [divider_width, 145], [divider_width, 144+159], [divider_width, 144]
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all obstacles
            fixed_obs.positions_lower_left = np.array([[-826 - 10, 0], [81, -250],
                                                       [-796, 724], [-796, 320], [-796, 0], [-221, -250],
                                                       [-846 - 10, 944], [-406, 944], [81, 742],
                                                       [731, 742], [-846 - 10, 808],
                                                       # vertical dividers that seperate desks and hallway
                                                       [-97-divider_width, 0], [-97-divider_width, 260], [-97-divider_width, 664]
                                                       ]) / 100.
            # 1: rectangular cube, 0: cylinder
            fixed_obs.shapes = np.array([1] * len(fixed_obs.sizes))

            # define human routes based on map
            human_flow.static_regions = np.array([[-650, -250, 115, 165], [-650, -250, 240, 290],
                                                  [-650, -250, 510, 550], [-650, -250, 650, 690]]) / 100.
            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {1: np.array([-60, 20, -400, -300]) / 100.,
                                  2: np.array([-200, -100, 115, 290]) / 100.,
                                  3: np.array([-60, 20, 0, 260]) / 100.,
                                  3.5: np.array([-60, 20, 300, 550]) / 100.,
                                  4: np.array([-200, -100, 520, 690]) / 100.,
                                  4.5: np.array([-650, -500, 550, 660]) / 100.,
                                  5: np.array([-60, 40, 563, 664]) / 100.,
                                  6: np.array([-650, -406, 840, 910]) / 100.,
                                  7: np.array([-221, 40, 790, 910]) / 100.,
                                  8: np.array([100, 600, 790, 910]) / 100.
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'correlated'

            human_flow.routes = [
                                # both human and robot's routes are straight lines
                                 [7, 5, 3, 1],
                                 [5, 3, 1],
                                 [3.5, 1],
                                # the human takes a turn and cross the robot
                                 [4, 5, 3, 1],
                                 [2, 3, 1],
                                # the human takes a turn and does not cross the robot
                                 [2, 3, 5, 7], [7, 5, 3, 2],
                                 [2, 3, 5, 4], [4, 5, 3, 2],
                                 [3.5, 5, 4],
                                 [7, 5, 4],
                                 ]
            human_flow.correlated_routes = [
                [[4.5, 5, 3, 1]],
                [[5, 3, 1]],
                [[3.5, 1]],
                [[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                 4.5, 4.5, 4.5, 4.5, 5, 3, 1]],

                [[5, 3, 1], [4.5, 5, 3, 1]],
                [[3.5, 1], [4.5, 5, 3, 1]],
                [[5, 3, 1], [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,4.5,5, 3, 1]],
                [[3.5, 1], [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,4.5, 5, 3, 1]],

                [[5, 3, 1],
                 [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                  4.5, 4.5, 4.5, 4.5, 5, 3, 1]],
                [[3.5, 1],
                 [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                  4.5, 4.5, 4.5, 4.5, 5, 3, 1]],
                [[5, 3, 1],
                 [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                  4.5, 4.5, 4.5, 4.5, 5, 3, 1]],
                [[3.5, 1],
                 [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                  4.5, 4.5, 4.5, 4.5, 5, 3, 1]],
                # [[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,4.5,5, 3, 1]]
            ]
   
        
        elif env.csl_workspace_type == 'lounge':
            fixed_obs.cylinder_radius = 0.5 # 0.45
            fixed_obs.cylinder_height = 1
            fixed_obs.sizes = np.array([[600, 10], [374, 474], # lower wall, left room
                                        [74, 242], [283, 339],  # sofas, right room
                                        [381, 646], [590, 10], [10, 646], [116, 14], # upper right room (hca lab), upper wall of cafe, left wall of cafe, lower left wall of cafe
                                        [90, 55], [90, 100], [75, 182], # trash cans, vending machine, rectangle table
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200], # treat circles as rectangles
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all rectangles, and centers of all cylinders
            fixed_obs.positions_lower_left = np.array([[-158.5, -146], [-532.5, -136], # lower wall, left room
                                                       [84.5, 48.5], [158.5, 0], # sofas, right room
                                                       [60.5, 509], [-529.5, 1155], [-539.5, 509], [-529.5, 509],# upper right room (hca lab), upper wall of cafe, left wall of cafe, lower left wall of cafe
                                                       [-404.5, 1100], [-314.5, 1090], [-14.5, 654],# trash cans, vending machine, rectangle table
                                                       [-367.5 - fixed_obs.cylinder_radius*100, 882 - fixed_obs.cylinder_radius*100],
                                                       [-98.5 - fixed_obs.cylinder_radius*100, 975 - fixed_obs.cylinder_radius*100],
                                                       [-119.5 - fixed_obs.cylinder_radius*100, 635 - fixed_obs.cylinder_radius*100] # lower left corner of 3 round tables
                                                      ]) / 100.
            # 1: rectangular cube, 0: cylinder
            fixed_obs.shapes = np.array([1] * 11 + [0] * 3)

            # define human routes based on map
            human_flow.static_regions = np.array([[-150, 60, 509, 1155],
                                                  [-520, -450, 509, 1155],
                                                  [74, 84.5+74, 48.5+242, 380], # sofa corner
                                                  [-158.5, 158.5, -120, -60], # near glass entrance door
                                                  ]) / 100.

            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {0: np.array([300, 400, -100, 0]) / 100.,
                                  1: np.array([-120, -20, -100, 0]) / 100.,
                                  2: np.array([-700, -400, 360, 470]) / 100.,
                                  3: np.array([-150, -20, 360, 470]) / 100.,
                                  3.5: np.array([-150, -20, 360, 600]) / 100.,
                                  4: np.array([400, 700, 360, 470]) / 100.,
                                  5: np.array([-450, -300, 500, 600]) / 100.,
                                  6: np.array([-200, -50, 650, 750]) / 100.,
                                  7: np.array([-300, -150, 900, 1000]) / 100.,
                                  8: np.array([-300, -150, 500, 600]) / 100.
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'independent'

            human_flow.routes = [
                                [2, 3, 4], [4, 3, 2],
                                [7, 8, 2], [7, 8, 4],
                                [7, 8, 3, 1],
                                [7, 8, 3, 1, 0],
                                [3, 1],
                                [3, 1, 0],
                                [3.5, 1],
                                [3.5, 1, 0],
            ]

            human_flow.correlated_routes = [
                                            [[7, 8, 3, 1], [3, 1]],
                                            [[7, 8, 3, 1], [3, 1, 0]],

                                            [[2, 3, 4], [4, 3, 2]],
                                            [[7, 5, 3], [2, 3, 1]]
                                            ]
        
        elif env.csl_workspace_type == 'corner':
            divider_width = 6
            fixed_obs.cylinder_radius = 0.5 # 0.45
            fixed_obs.cylinder_height = 0.7
            # only includes the first 3 lines of workstation from bottom
            # [width, height] of all obstacles
            
            fixed_obs.sizes = np.array([[10, 1300],
                                        [700, 360], [700, 360], [140, 250],
                                        [1100, 250], 
                                        # 两个桌子
                                        [450, 200], [450, 200],
                                        # vertical dividers that seperate desks and hallway
                                        [divider_width, 360], [divider_width, 360],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200]
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all obstacles
            fixed_obs.positions_lower_left = np.array([[590, -250],
                                                       [-800, -20], [-800, 500], [-240, -250],
                                                       [-500, 1040], 
                                                       # 两个桌子
                                                       [150, 650], [150, 150],
                                                       # vertical dividers that seperate desks and hallway
                                                       [-97-divider_width, 0], [-97-divider_width, 480],
                                                       [200 - fixed_obs.cylinder_radius*100, 500 - fixed_obs.cylinder_radius*100],
                                                       [200 - fixed_obs.cylinder_radius*100, 0 - fixed_obs.cylinder_radius*100]
                                                       ]) / 100.
            fixed_obs.shapes = np.array([1] * 9 + [0] * 2)

            # define human routes based on map
            human_flow.static_regions = np.array([[310, 570, -100, 100], [80, 150, 500, 600], 
                                                  [310, 570, 350, 600],
                                                  [-80, -20, 130, 300]
                                                  ]) / 100.
            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {0: np.array([400, 500, -200, -100]) / 100.,
                                  1: np.array([200, 300, -250, -150]) / 100.,
                                  1.5: np.array([50, 200, -250, -100]) / 100.,
                                  2: np.array([0, 100, 0, 100]) / 100.,
                                  3: np.array([-50, 0, 350, 450]) / 100.,
                                  3.5: np.array([0, 50, 600, 700]) / 100.,
                                  4: np.array([-20, 80, 800, 900]) / 100.,
                                  5: np.array([300, 500, 890, 950]) / 100.,
                                  6: np.array([-400, -200, 900, 1010]) / 100.,
                                  7: np.array([-640, -570, 370, 470]) / 100.,
                                  7.5: np.array([-570, -500, 370, 430]) / 100.,
                                  7.75: np.array([-710, -640, 370, 470]) / 100.,
                                  8: np.array([-750, -550, 1000, 1200]) / 100.,
                                  8.5: np.array([-600, -500, 900, 1000]) / 100.,
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'correlated'

            human_flow.routes = [
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
            
            human_flow.correlated_routes = [
                #corner
                [[7, 7.5, 3, 2, 1], [8, 6, 4, 2, 1.5]],
                [[3, 4, 6, 5], [3, 3, 3, 3.5, 6, 4, 3, 2, 1.5]],
                # 2 people
                [[1.5, 4], [4, 2, 1.5]],
                [[1.5, 3, 7],[6, 4.5, 3, 7.5]],
                # 2 people corner
                [[7, 7.5, 3, 2, 1], [7.5, 3, 2, 1], 
                 [8.5, 4, 2], [8, 8.5, 4, 2]],
                [[7, 7.5, 3, 3.5, 4, 5], [7.5, 3, 3.5, 4, 5], 
                 [8, 8.5, 6, 5], [1, 1.5, 4, 6]],
                # many people
                [[7, 3, 3.5, 4, 5], [7.5, 3, 3.5, 4, 5],
                 [7.75, 3, 2, 1, 0]]
                
            ]

        elif env.csl_workspace_type == 'corridor':
            divider_width = 6
            fixed_obs.sizes = np.array([# walls
                                        [10, 1200], [400, 250],
                                        [10, 200], [10, 200], 
                                        [650, 10], [100, 10],
                                        [10, 1000], [10, 300], [250, 150], [150, 300],
                                        # vertical dividers that seperate desks and hallway
                                        [30, divider_width], [30, divider_width],
                                        [30, divider_width], [30, divider_width]
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all obstacles
            fixed_obs.positions_lower_left = np.array([# walls
                                                       [-100, -100], [-500, -100],
                                                       [-200, 1100], [740, 1100],
                                                       [90, 1100], [-200, 1100],
                                                       [90, -300], [90, 800], [-150, -400], [-500, -400],
                                                       # vertical dividers that seperate desks and hallway
                                                       [-100, -100], [100-30, -100],
                                                       [-350, -400], [-150-30, -400]                                                
                                                       ]) / 100.
            # 1: rectangular cube, 0: cylinder
            fixed_obs.shapes = np.array([1] * len(fixed_obs.sizes))

            # define human routes based on map
            
            human_flow.static_regions = np.array([[-180, -70, 1180, 1300], [35, 100, 650, 850],
                                                  ]) / 100.
            
            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {0: np.array([400, 600, 1100, 1250]) / 100.,
                                  1: np.array([150, 300, 1150, 1250]) / 100.,
                                  1.5: np.array([-25, 25, 1050, 1130]) / 100.,
                                  2: np.array([-75, 0, 850, 930]) / 100.,
                                  3: np.array([-0, 75, 550, 600]) / 100.,
                                  4: np.array([-75, 50, 300, 400]) / 100.,
                                  5: np.array([-0, 75, 100, 200]) / 100.,
                                  6: np.array([-75, 75, -100, 0]) / 100.,
                                  7: np.array([-200, -50, -230, -150]) / 100.,
                                  7.5: np.array([-300, -200, -400, -270]) / 100.
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'correlated'

            human_flow.routes = [
                                # human goes right
                                 [0, 1, 1.5, 2, 4, 6, 7],
                                 [2, 4, 6, 7, 8],
                                # human goes left
                                 [0, 1, 1.5, 3, 5, 7],
                                # human goes left and then right
                                 [1, 1.5, 3, 4, 6, 7],
                                 ]
            human_flow.correlated_routes = [
                # human goes right,towards robot
                [[1, 1.5, 2, 4, 6, 7, 7.5], [0, 1, 1.5, 2, 4, 6, 7, 7.5],
                 [0, 0, 0, 1, 1.5, 2, 4, 6, 7, 7.5]],
                # human goes left, then right
                [[0, 1, 1.5, 2, 4, 6, 7]],
                [[1, 1.5, 3, 4, 6, 7], [0, 1, 1.5, 2, 4, 6, 7]],
                # human walk torwards
                [[1, 1.5, 2, 4, 6, 7, 7.5], [0, 1, 1.5, 2, 4, 6, 7], 
                 [7, 6, 5, 3, 1.5]],
            ]

        elif env.csl_workspace_type == 'square':
            #fixed_obs.sizes = np.array([[10, 500]]) / 100.
            fixed_obs.sizes = np.array([[10, 500], [600, 10], [10, 300],
                                        [200, 300], [90, 400], [190, 100],
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all obstacles
            #fixed_obs.positions_lower_left = np.array([[-300, -100]]) / 100.
            fixed_obs.positions_lower_left = np.array([[-300, -100], [-300, 400], [290, 100],
                                                       [300, 100], [-300, -100], [-300, 300]
                                                       ]) / 100.
            fixed_obs.shapes = np.array([1] * 6)

            # define human routes based on map
            human_flow.static_regions = np.array([[310, 570, -100, 100], [310, 570, 350, 600],
                                                  [-80, 150, 150,300], [-80, 150, 500, 600]
                                                  ]) / 100.
            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {1: np.array([300, 400, -220, -100]) / 100.,
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
            human_flow.route_type = 'correlated'

            human_flow.routes = [
                                
                                 ]
            
            human_flow.correlated_routes = [
                
            ]

        else:
            raise ValueError("Unknown csl_workspace_type")
        assert len(fixed_obs.sizes) == len(fixed_obs.positions_lower_left)

        # change the static obs information based on the fixed_obs above
        sim.static_obs_num = len(fixed_obs.sizes)
        sim.static_obs_num_range = 0

        # make sure each route has a start region and at least one goal region
        for route in human_flow.routes:
            assert len(route) >= 2

        # adjust the human_num to prevent errors for correlated routes
        if human_flow.route_type == 'correlated':
            sim.human_num = sim.static_human_num + max(len(sublist) for sublist in human_flow.correlated_routes)
            sim.human_num_range = 0

    # for circle crossing sceanrio, humans start & goals are always sampled randomly
    else:
        human_flow.route_type = 'independent'


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
        robot.radius = 0.2
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
    robot.w_max = 1
    robot.w_min = -1
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

    # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
    if env.csl_workspace_type == 'hallway':
        robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                         2: np.array([-0.3, 0.3, 5.5, 6]),
                         3: np.array([-5, -4, 5.5, 6.5]),
                         }
        # short-distance navigation
        robot.routes = [[1, 3]
                        ]
    elif env.csl_workspace_type == 'lounge':
        robot.regions = {1: np.array([-0.5, 0.5, -0.3, 0.3]),
                         2: np.array([-4, -3, 9, 10]),
                         3: np.array([-1.5, -0.5, 7.5, 8.5]),
                         }
        # short-distance navigation
        robot.routes = [[1, 3]
                        ]
    elif env.csl_workspace_type == 'corner':
        robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                         2: np.array([-0.3, 0.3, 5.5, 6]),
                         3: np.array([-4, -3, 9, 10]),
                         }
        # short-distance navigation
        robot.routes = [[1, 3]
                        ]
    elif env.csl_workspace_type == 'corridor':
        robot.regions = {1: np.array([-0.2, 0.2, -0.3, 0.3]),
                         2: np.array([-0.3, 0.3, 5.5, 6]),
                         3: np.array([4, 6, 13, 15]),
                         }
        # short-distance navigation
        robot.routes = [[1, 3]
                        ]

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
    humans.radius = 0.25 # radius of each human
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
    humans.dynamic_activity = ['walking', 'running', 'carrying']
    humans.static_activity = ['stationary', 'talking']

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
    # todo: change this
    training.output_dir = 'data/ours_RH_HH_hallwayEnv_new'  # the saving directory for train.py
    # resume training from an existing checkpoint or not
    # none: train RL from scratch, rl: load a RL weight
    training.resume = 'none'
    # if resume != 'none', load from the following checkpoint
    training.load_path = 'trained_models/ours_HH_RH_randEnv/checkpoints/237800.pt'
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
