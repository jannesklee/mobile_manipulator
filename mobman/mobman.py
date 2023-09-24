import numpy as np
from scipy.linalg import logm, pinv
import sys
sys.path.append('/home/jklee/src/ModernRobotics/packages/Python/')
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

def array_output(FlatTrajectory):
    gripper = FlatTrajectory[12]
    traj =  np.identity(4)
    traj[0:3,0:3] = np.reshape(FlatTrajectory[0:9], (3,3))
    traj[0:3,3] = FlatTrajectory[9:12]

    return traj, gripper



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
    method = 5 # cubic time scaling

    # A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
    tf1 = 10 # 10s
    N = tf1*100*k
    InitialToStandoff = np.array(mr.ScrewTrajectory(Tse_init, Tce_standoff, tf1, N, method))
    InitialToStandoffFlat = flatten_output(InitialToStandoff, 0.)
    # A trajectory to move the gripper down to the grasp position. -> 4s
    tf2 = 4 # 4s
    N = tf2*100*k
    StandoffToGrasp = np.array(mr.ScrewTrajectory(Tce_standoff, Tce_grasp, tf2, N, method))
    StandoffToGraspFlat = flatten_output(StandoffToGrasp, 0.)

    # Closing of the gripper -> 1s
    GripperSteps = 100*k # =s
    CloseGripperFlat = np.zeros((GripperSteps,13))
    for i, gripper in enumerate(np.linspace(0,1,GripperSteps)):
        CloseGripperFlat[i,:] = StandoffToGraspFlat[-1,:]
        CloseGripperFlat[i,12] = gripper

    # A trajectory to move the gripper back up to the "standoff" configuration. -> 4s
    tf3 = 4 # 4s
    N = tf3*100*k
    GraspToStandoff = np.array(mr.ScrewTrajectory(Tce_grasp, Tce_standoff, tf3, N, method))
    GraspToStandoffFlat = flatten_output(GraspToStandoff,1.)

    # A trajectory to move the gripper to a "standoff" configuration above the final configuration.
    # get same relative position as at the first standoff -> 10s
    Tsc_rel = np.matmul(np.linalg.inv(Tsc_init), Tce_standoff)
    Tce_finalstandoff = np.matmul(Tsc_final,Tsc_rel)
    tf4 = 10 # 4s
    N = tf4*100*k
    StandoffToStandoff = np.array(mr.ScrewTrajectory(Tce_standoff, Tce_finalstandoff, tf4, N, method))
    StandoffToStandoffFlat = flatten_output(StandoffToStandoff, 1.)

    #A trajectory to move the gripper to the final configuration of the object. -> 4s
    Tsgrasp_rel = np.matmul(np.linalg.inv(Tsc_init), Tce_grasp)
    Tce_finalgrasp = np.matmul(Tsc_final,Tsgrasp_rel)
    tf5 = 4
    N = tf5*100*k
    StandoffToGraspFinal = np.array(mr.ScrewTrajectory(Tce_finalstandoff, Tce_finalgrasp, tf5, N, method))
    StandoffToGraspFinalFlat = flatten_output(StandoffToGraspFinal, 1.)

    # Opening of the gripper -> 1s
    OpenGripperFlat = np.zeros((GripperSteps,13))
    for i, gripper in enumerate(np.linspace(1,0,GripperSteps)):
        OpenGripperFlat[i,:] = StandoffToGraspFinalFlat[-1,:]
        OpenGripperFlat[i,12] = gripper

    #A trajectory to move the gripper back to the "standoff" configuration. -> 4s
    tf6 = 4
    N = tf6*100*k
    StandoffToStandoffFinal = np.array(mr.ScrewTrajectory(Tce_finalgrasp, Tce_finalstandoff, tf6, N, method))
    StandoffToStandoffFinalFlat = flatten_output(StandoffToStandoffFinal, 0.)

    # whole trajectory -> 38s
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
    VdM = logm(np.matmul(np.linalg.inv(Tse_d),Tse_d_next))/dt
    Vd = np.array([VdM[1,2],VdM[0,2],VdM[0,1],VdM[0,3],VdM[1,3],VdM[2,3]])
    X_errM = logm(np.matmul(np.linalg.inv(Tse),Tse_d))
    X_err = np.array([X_errM[1,2],X_errM[0,2],X_errM[0,1],X_errM[0,3],X_errM[1,3],X_errM[2,3]])
    AdXXd = mr.Adjoint(np.matmul(np.linalg.inv(Tse), Tse_d))
    #AdX = np.array([AdXXd[1,2],AdXXd[0,2],AdXXd[0,1],AdXXd[0,3],AdXXd[1,3],AdXXd[2,3]])
    V = np.matmul(AdXXd, Vd) + np.matmul(Kp,X_err) + np.matmul(Ki,X_err)*dt

    return V, X_err

def get_speeds(V, Tse, state):
    x = state[0]
    y = state[1]
    varphi = state[2]

    # feedforward
    Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Tb0 = np.array([[1, 0., 0., 0.1662],[0., 1, 0., 0.],[0., 0., 1, 0.0026],[0., 0., 0., 1]])

    # calculate Jarm
    B =  np.array([
        [0.0 , 0.0 , 1.0, 0.0    , 0.033, 0.0],        
        [0.0 , -1.0, 0.0, -0.5076, 0.0  , 0.0],     
        [0.0 , -1.0, 0. , -0.3526, 0.0  , 0.00],     
        [0.0 , -1.0, 0. , -0.2176, 0.0  , 0.00],     
        [0.0 , 0.0 , 1. , 0.0    , 0.0  , 0.00]]).T
    thetalist = np.array(state[3:3+5]).T
    Jarm = mr.JacobianBody(B, thetalist)

    # calculate Jbase
    l = 0.47*0.5
    w = 0.3*0.5
    r = 0.0475
    F = 0.25*r*np.array([
        [-1./(l+w), 1./(l+w), 1./(l+w), -1./(l+w)],
        [1        , 1       , 1       , 1        ],
        [-1       , 1       , -1      , 1        ]
    ])

    Adj = mr.Adjoint(np.matmul(np.linalg.inv(Tse), np.linalg.inv(Tb0)))
    F6 = np.vstack([np.zeros(4),np.zeros(4),F,np.zeros(4)])
    Jbase = np.matmul(Adj,F6)

    # combine in one jacobian
    Je = np.hstack([Jbase, Jarm])

    u_and_theta_dot = np.matmul(pinv(Je, atol=1e-4),V)

    return u_and_theta_dot

def get_endeffector(state):
    x = state[0]
    y = state[1]
    varphi = state[2]

    Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Tb0 = np.array([[1, 0., 0., 0.1662],[0., 1, 0., 0.],[0., 0., 1, 0.0026],[0., 0., 0., 1]])
    M0e = np.array([
        [1., 0., 0., 0.033],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.6546],
        [0., 0., 0., 1.]])
    B =  np.array([
        [0.0 , 0.0 , 1.0, 0.0    , 0.033, 0.0],        
        [0.0 , -1.0, 0.0, -0.5076, 0.0  , 0.0],     
        [0.0 , -1.0, 0. , -0.3526, 0.0  , 0.00],     
        [0.0 , -1.0, 0. , -0.2176, 0.0  , 0.00],     
        [0.0 , 0.0 , 1. , 0.0    , 0.0  , 0.00]]).T
    thetalist = state[2:2+5]
    X = mr.FKinBody(M0e, B, thetalist)
    return X

# input for trajectory generator
Tsc_init = np.array([[1, 0, 0, 1],[0, 1, 0, 0],[0, 0, 1, 0.025],[0, 0, 0, 1]])
Tsc_final = np.array([[0, 1, 0, 0],[-1, 0, 0, -1],[0, 0, 1, 0.025],[0, 0, 0, 1]])
Tse_init = np.array([[1, 0, 0, 0.033],[0, 1, 0, 0],[0, 0, 1, 0.6546],[0, 0, 0, 1]])
xi = -3./4.*np.pi # angle for putting down or grabbing cube
Tce_standoff = np.array([[np.cos(xi), 0, -np.sin(xi), 1],[0, 1, 0, 0],[np.sin(xi), 0, np.cos(xi), 0.125],[0, 0, 0, 1]])
Tce_grasp = Tce_standoff.copy()
Tce_grasp[0:3,3] = Tsc_init[0:3,3]
k = 1

Trajectory = trajectory_generator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k)
np.savetxt('trajectory.csv', Trajectory, delimiter=', ')

##### configuration calculated feedback
state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# calc dt from k
dt = 0.01
Kp = np.zeros((6,6))
#Kp = np.identity(6)
Ki = np.zeros((6,6))
speedlimits = 100.0

# initial start of end-effector (normally not true)
(Tse, _) = array_output(Trajectory[0,:])

print(Tse)

# iterate over trajectory
state_save = []
Xerr_save = []
for i in range(np.shape(Trajectory)[0]-1):
    (Tse_d, gripper) = array_output(Trajectory[i,:])
    (Tse_d_next, _) = array_output(Trajectory[i+1,:])

    V, Xerr = feedback_control(Tse, Tse_d, Tse_d_next, Kp, Ki, dt)
    uthetadot = get_speeds(V, Tse, state)
    print(uthetadot)
    
    state = next_state(state, uthetadot, dt, speedlimits)
    Tse = get_endeffector(state) # real new position
    

    if (i % k) == 0:
        state_save.append(np.hstack((state, gripper)))
        Xerr_save.append(Xerr)
        pass

np.savetxt('states.csv', np.array(state_save), delimiter=', ')
np.savetxt('Xerr.csv', np.array(Xerr_save), delimiter=', ')


# make a test of this
#x = 0.0
#y = 0.0
#varphi = 0.0
#thetalist = np.array([0.0,0.0,0.2,-1.6,0.])
#Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
#Tb0 = np.array([[1, 0., 0., 0.1662],[0., 1, 0., 0.],[0., 0., 1, 0.0026],[0., 0., 0., 1]])
#M0e = np.array([
#    [1., 0., 0., 0.033],
#    [0., 1., 0., 0.],
#    [0., 0., 1., 0.6546],
#    [0., 0., 0., 1.]])
#B =  np.array([
#    [0.0 , 0.0 , 1.0, 0.0    , 0.033, 0.0],        
#    [0.0 , -1.0, 0.0, -0.5076, 0.0  , 0.0],     
#    [0.0 , -1.0, 0. , -0.3526, 0.0  , 0.00],     
#    [0.0 , -1.0, 0. , -0.2176, 0.0  , 0.00],     
#    [0.0 , 0.0 , 1. , 0.0    , 0.0  , 0.00]]).T
#Xgenau = mr.FKinBody(M0e, B, thetalist)
#state = np.hstack((np.array([x,y,varphi]).T,thetalist,np.array([0.,0.,0.,0.]).T))
#
#X = np.array([
#    [0.170   , 0.0 , 0.985   , 0.387  ],
#    [0.0   , 1.0 , 0.0   , 0.0  ], 
#    [-0.985  , 0.0 , 0.17   , 0.57  ],
#    [0.0   , 0.0 , 0.0   , 1.0  ]])
#Xd = np.array([
#    [0.0   , 0.0 , 1.0   , 0.5  ],
#    [0.0   , 1.0 , 0.0   , 0.0  ], 
#    [-1.0  , 0.0 , 0.0   , 0.5  ],
#    [0.0   , 0.0 , 0.0   , 1.0  ]])
#Xd_next = np.array([
#    [0.0   , 0.0 , 1.0   , 0.6  ],
#    [0.0   , 1.0 , 0.0   , 0.0  ], 
#    [-1.0  , 0.0 , 0.0   , 0.3  ],
#    [0.0   , 0.0 , 0.0   , 1.0  ]])
##Kp = np.zeros((6,6))
#Kp = np.identity(6)
#Ki = np.zeros((6,6))
#dt = 0.01
#
#(V, Xerr) = feedback_control(X, Xd, Xd_next, Kp, Ki, dt)
#u = get_speeds(V, X, state)
#print(u)
