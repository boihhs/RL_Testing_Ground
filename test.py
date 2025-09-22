from Get_IL_Data.get_ground_truth_pos import getGroundTurthPositions

robot_data_path = "/home/leo-benaharon/Desktop/HumanPose/GMR/path_to_save_robot_data.pkl"
cfg_file = "/home/leo-benaharon/Desktop/HumanPose/RL_Testing_Ground/RL_Algos/PPO.yaml"

e = getGroundTurthPositions(robot_data_path, cfg_file, .1)
e.run()
# print(e.body_positions)