from legged_gym.envs.base.juggle_robot_config import JuggleRobotCfg, JuggleRobotCfgPPO

class JuggleCfg( JuggleRobotCfg ):
    class env( JuggleRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 10



    class init_state( JuggleRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'R_hip_joint': 0.,
            'R_hip2_joint': 0.,
            'R_thigh_joint': 0,
            'R_calf_joint': 0.,
            'R_toe_joint': 0.,
            'L_hip_joint': 0.,
            'L_hip2_joint': 0.,
            'L_thigh_joint': 0.,
            'L_calf_joint': 0.,
            'L_toe_joint': 0.,
        }

    class control(JuggleRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_joint': 200.0, 'hip2_joint': 200.,  'thigh_joint': 200., 'calf_joint': 200.,  'toe_joint': 40.}  # [N*m/rad]
        
        damping = {'hip_joint': 10, 'hip2_joint': 10, 'thigh_joint': 10.,'calf_joint': 10, 'toe_joint': 10}
        # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( JuggleRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/juggle/urdf/juggle.urdf'
        name = "juggle"
        foot_name = 'toe'
        #terminate_after_contacts_on = ['base_link']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( JuggleRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 200.
        only_positive_rewards = False
        class scales( JuggleRobotCfg.rewards.scales ):
            termination = 0


class JuggleCfgPPO( JuggleRobotCfgPPO ):
    
    class runner( JuggleRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'juggle'

    class algorithm( JuggleRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  