import numpy as np

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

state = np.array([
    0.000000,   0.000000,   0.000000,   
    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
    0.000000,   0.000000,   0.000000,   0.000000])
#speeds = np.array([
#    10., 10., 10., 10., 
#    0., 0., 0., 0., 0.])
#speeds = np.array([
#    -10., 10., -10., 10., 
#    0., 0., 0., 0., 0.])
speeds = np.array([
    -10., 10., 10., -10., 
    0., 0., 0., 0., 0.])
dt = 0.01

for i in range(100):
    state = next_state(state, speeds, dt, 100)

print(state)
