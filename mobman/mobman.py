import numpy as np
from scipy.linalg import logm
import modern_robotics as mr

def next_state(state, speeds, dt, speedlimits):
    q = state[0:3]
    joint_state = state[3:3+5]
    wheel_state = state[3+5:12]

    wheel_speeds = speeds[0:4]
    joint_speeds = speeds[4:4+5]

    new_joint_state = joint_state + dt*joint_speeds
    new_wheel_state = wheel_state + dt*wheel_speeds

    # odometry
    l = 0.47*0.5
    w = 0.3*0.5
    r = 0.0475

    F = 0.25*r*np.array([
        [-1./(l+w), 1./(l+w), 1./(l+w), -1./(l+w)],
        [1        , 1       , 1       , 1        ],
        [-1       , 1       , -1      , 1        ]
    ])
    Vb = np.matmul(F,dt*wheel_speeds)

    if Vb[0] == 0:
        dq_b = np.array([0, Vb[1], Vb[2]]).T
    else:
        dq_b = np.array([
            Vb[0], 
            Vb[1]*np.sin(Vb[0])+Vb[2]*(np.cos(Vb[0]-1))/Vb[0],
            Vb[2]*np.sin(Vb[0])+Vb[1]*(1-np.cos(Vb[0]))/Vb[0]]).T

    dq = np.matmul(np.array([
        [1, 0, 0], 
        [0, np.cos(q[0]), -np.sin(q[0])],
        [0, np.sin(q[0]), np.cos(q[0])]]), dq_b)

    new_chassis_state = q + dq

    return np.append(np.append(new_chassis_state, new_joint_state), new_wheel_state)

def flatten_output(TrajectoryIn, gripper):
    Trajectory = np.zeros((len(TrajectoryIn),13))
    for i in range(len(TrajectoryIn)):
        Trajectory[i,:9] = TrajectoryIn[i,0:3,0:3].flatten()
        Trajectory[i,9:12] = TrajectoryIn[i,0:3,3].flatten()
        Trajectory[i,12] = gripper

    return Trajectory

def trajectory_generator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k):
    """
    Input
    The initial configuration of the end-effector in the reference trajectory: Tse_init
    The cube's initial configuration: Tsc_init
    The cube's desired final configuration: Tsc_final
    The end-effector's configuration relative to the cube when it is grasping the cube: Tce_grasp
    The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube: Tce_standoff. This specifies the configuration of the end-effector {e} relative to the cube frame {c} before lowering to the grasp configuration Tce_grasp
    The number of trajectory reference configurations per 0.01 seconds: k. value of 1 or greater. Although your final animation will be based on snapshots separated by 0.01 seconds in time, the points of your reference trajectory (and your controller servo cycle) can be at a higher frequency. For example, if you want your controller to operate at 1000 Hz, you should choose k = 10 {\displaystyle k=10} (10 reference configurations, and 10 feedback servo cycles, per 0.01 seconds). It is fine to choose k = 1 {\displaystyle k=1} if you'd like to keep things simple
    """

    # likely use ScrewTrajectory or CartesianTrajectory
    Tf = k
    method = 5 # cubic time scaling

    # A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
    InitialToStandoff = np.array(mr.ScrewTrajectory(Tse_init, Tce_standoff, Tf, k, method))
    InitialToStandoffFlat = flatten_output(InitialToStandoff, 0.)
    # A trajectory to move the gripper down to the grasp position.
    StandoffToGrasp = np.array(mr.ScrewTrajectory(Tce_standoff, Tce_grasp, Tf, k, method))
    StandoffToGraspFlat = flatten_output(StandoffToGrasp, 0.)

    # Closing of the gripper
    GripperSteps = 50
    CloseGripperFlat = np.zeros((GripperSteps,13))
    for i, gripper in enumerate(np.linspace(0,1,GripperSteps)):
        CloseGripperFlat[i,:] = StandoffToGraspFlat[-1,:]
        CloseGripperFlat[i,12] = gripper

    # A trajectory to move the gripper back up to the "standoff" configuration.
    GraspToStandoff = np.array(mr.ScrewTrajectory(Tce_grasp, Tce_standoff, Tf, k, method))
    GraspToStandoffFlat = flatten_output(GraspToStandoff,1.)

    # A trajectory to move the gripper to a "standoff" configuration above the final configuration.
    # get same relative position as at the first standoff
    Tsc_rel = np.matmul(np.linalg.inv(Tsc_init), Tce_standoff)
    Tce_finalstandoff = np.matmul(Tsc_final,Tsc_rel)
    StandoffToStandoff = np.array(mr.ScrewTrajectory(Tce_standoff, Tce_finalstandoff, Tf, k, method))
    StandoffToStandoffFlat = flatten_output(StandoffToStandoff, 1.)

    #A trajectory to move the gripper to the final configuration of the object.
    Tsgrasp_rel = np.matmul(np.linalg.inv(Tsc_init), Tce_grasp)
    Tce_finalgrasp = np.matmul(Tsc_final,Tsgrasp_rel)
    StandoffToGraspFinal = np.array(mr.ScrewTrajectory(Tce_finalstandoff, Tce_finalgrasp, Tf, k, method))
    StandoffToGraspFinalFlat = flatten_output(StandoffToGraspFinal, 1.)

    # Opening of the gripper
    OpenGripperFlat = np.zeros((GripperSteps,13))
    for i, gripper in enumerate(np.linspace(1,0,GripperSteps)):
        OpenGripperFlat[i,:] = StandoffToGraspFinalFlat[-1,:]
        OpenGripperFlat[i,12] = gripper

    #A trajectory to move the gripper back to the "standoff" configuration.
    StandoffToStandoffFinal = np.array(mr.ScrewTrajectory(Tce_finalgrasp, Tce_finalstandoff, Tf, k, method))
    StandoffToStandoffFinalFlat = flatten_output(StandoffToStandoffFinal, 0.)

    Trajectory = np.concatenate((
        InitialToStandoffFlat,
        StandoffToGraspFlat,
        CloseGripperFlat,
        GraspToStandoffFlat,
        StandoffToStandoffFlat,
        StandoffToGraspFinalFlat,
        OpenGripperFlat,
        StandoffToStandoffFinalFlat), axis=0)

    return Trajectory

def feedback_control(Tse, Tse_d, Tse_d_next, Kp, Ki, dt):

    Vd = mr.MatrixLog6(np.linalg.inv(Tse_d), Ts_d_next)
    X_err = mr.MatrixLog6(np.matmul(np.linalg.inv(X),X))

    np.matmul(mr.Adjoint(np.matmul(np.linalg.inv(Tse), Tse_d)), Vd) + \
        np.matmul(Kp,Xerr) + np.matmul(Ki, Xerr)*dt

    pass

def test_joint_limits():
    #With these joint limits, you could write a function called testJointLimits to return a list of joint limits that are violated given the robot arm's configuration θ {\displaystyle \theta }. 
    pass

# input for trajectory generator
#Tsc_init = np.array([[1, 0, 0, 1],[0, 1, 0, 0],[0, 0, 1, 0.025],[0, 0, 0, 1]])
#Tsc_final = np.array([[0, 1, 0, 0],[-1, 0, 0, -1],[0, 0, 1, 0.025],[0, 0, 0, 1]])
#
#Tce_grasp = Tsc_init
#Tse_init = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0.5],[0, 0, 0, 1]])
#varphi = -3./4.*np.pi
#Tce_standoff = np.array([[np.cos(varphi), 0, -np.sin(varphi), 1],[0, 1, 0, 0],[np.sin(varphi), 0, np.cos(varphi), 0.125],[0, 0, 0, 1]])
#Tce_grasp = Tce_standoff.copy()
#Tce_grasp[0:3,3] = Tsc_init[0:3,3]
#
#Trajectory = trajectory_generator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, 100)
#np.savetxt('test.csv', Trajectory, delimiter=', ')


#Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0, x],[np.sin(varphi), np.cos(varphi), 0, y],[0, 0, 1, 0.0963],[0, 0, 0, 1]])
#Tb0 = np.array([[1, 0, 0, 0.1662],[0, 1, 0, 0],[0, 0, 1, 0.0026],[0, 0, 0, 1]])
#Tse_init = np.array([[np.cos(varphi), -np.sin(varphi), 0, x],[np.sin(varphi), np.cos(varphi), 0, y],[0, 0, 1, 0.0963],[0, 0, 0, 1]])
#Tse_init = np.array([[np.cos(varphi), -np.sin(varphi), 0, x],[np.sin(varphi), np.cos(varphi), 0, y],[0, 0, 1, 0.0963],[0, 0, 0, 1]])

X = np.array([
    [0.170 , 0.0 , 0.985 , 0.387],
    [0.0   , 1.0 , 0.0   , 0.0  ], 
    [-0.985, 0.0 , 0.170 , 0.570],
    [0.0   , 0.0 , 0.0   , 1.0  ]])
Xd = np.array([
    [0.0   , 0.0 , 1.0   , 0.5  ],
    [0.0   , 1.0 , 0.0   , 0.0  ], 
    [-1.0  , 0.0 , 0.0   , 0.5  ],
    [0.0   , 0.0 , 0.0   , 1.0  ]])
Xd_next = np.array([
    [0.0   , 0.0 , 1.0   , 0.6  ],
    [0.0   , 1.0 , 0.0   , 0.0  ], 
    [-1.0  , 0.0 , 0.0   , 0.3  ],
    [0.0   , 0.0 , 0.0   , 1.0  ]])
Kp = np.zeros((4,4))
Ki = np.zeros((4,4))
dt = 0.01

VdM = logm(np.matmul(np.linalg.inv(Xd),Xd_next))/dt
Vd = np.array([VdM[1,2],VdM[0,2],VdM[0,1],VdM[0,3],VdM[1,3],VdM[2,3]])
first = np.matmul(mr.Adjoint(np.matmul(np.linalg.inv(X),Xd)), Vd)
X_errM = logm(np.matmul(np.linalg.inv(X),Xd))
X_err = np.array([X_errM[1,2],X_errM[0,2],X_errM[0,1],X_errM[0,3],X_errM[1,3],X_errM[2,3]])

# calc forward kinematics to get Jacobian
M = [[1,0,0,3.732],[0,1,0,0],[0,0,1,2.732],[0,0,0,1]]
B = [[0,0,0,0,0,0],
 [0,1,1,1,0,0],
 [1,0,0,0,0,1],
 [0,2.73,3.73,2,0,0],
 [2.73,0,0,0,0,0],
 [0,-2.73,-1,0,1,0]]
mr.FKinBody(M, B, [-np.pi/2, np.pi/2, np.pi/3, -np.pi/4, 1, np.pi/6])