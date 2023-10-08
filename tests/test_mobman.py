from mobman.mobman import *
import pytest
import numpy as np

@pytest.fixture
def initial_state():
    return np.array([
        0.000000,   0.000000,   0.000000,   
        0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
        0.000000,   0.000000,   0.000000,   0.000000])

@pytest.mark.parametrize('speeds, expected', [
    (np.array([10., 10., 10., 10., 0., 0., 0., 0., 0.]), np.array([0, 0.475, 0])),
    (np.array([-10., 10., -10., 10., 0., 0., 0., 0., 0.]), np.array([0, 0, 0.475])),
    (np.array([-10., 10., 10., -10., 0., 0., 0., 0., 0.]), np.array([1.234, 0, 0])),
    (np.array([0.5, 0.5, 0.5, 0.5, 0., 0., 0., 0., 0.]), np.array([0., 0.475e-1/2, 0])),
])
def test_next_state(initial_state, speeds, expected):
    state = initial_state
    dt = 0.01

    for i in range(100):
        state = next_state(state, speeds, dt, 100)

    print(state)

    assert np.allclose(state[:3], expected[:3], atol=1e-3)

def test_feedforward_control():
    x = 0.0
    y = 0.0
    varphi = 0.0
    thetalist = np.array([0.,0.,0.2,-1.6,0.])
    state = np.hstack((np.array([varphi,x,y]).T,thetalist,np.array([0.,0.,0.,0.]).T))

    Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Tb0 = np.array([[1., 0., 0., 0.1662],[0., 1., 0., 0.],[0., 0., 1., 0.0026],[0., 0., 0., 1.]])
    M0e = np.array([
        [1., 0., 0., 0.033],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.6546],
        [0., 0., 0., 1.]])
    B = get_B()
    (_, T0e) = get_endeffector(state)
    X = np.matmul(np.matmul(Tsb,Tb0),T0e)

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
    Kp = np.zeros((6,6))
    Ki = np.zeros((6,6))
    dt = 0.01

    (V, Xerr) = feedback_control(X, Xd, Xd_next, Kp, Ki, dt)

    V_expected = np.array([0., 0., 0., 21.409, 0., 6.455])
    Xerr_expected = np.array([0., 0.171, 0., 0.080, 0., 0.107])
    assert np.allclose(V, V_expected, atol=1e-3)
    assert np.allclose(Xerr, Xerr_expected, atol=1e-3)

    (u, Je) = get_speeds(V, state)
    Je_expected = np.array([
        [ 0.030, -0.030, -0.030,  0.030, -0.985,  0.000,  0.000,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.000,  0.000, -1.000, -1.000, -1.000,  0.000],
        [-0.005,  0.005,  0.005, -0.005,  0.170,  0.000,  0.000,  0.000,  1.000],
        [ 0.002,  0.002,  0.002,  0.002,  0.000, -0.240, -0.214, -0.218,  0.000],
        [-0.024,  0.024,  0.000,  0.000,  0.221,  0.000,  0.000,  0.000,  0.000],
        [ 0.012,  0.012,  0.012,  0.012,  0.000, -0.288, -0.135,  0.000,  0.000]])
    u_expected = np.array([157.2, 157.2, 157.2, 157.2, 0.0, -652.9, 1398.6, -745.7, 0.0])
    assert np.allclose(Je, Je_expected, atol=1e-3)
    assert np.allclose(u, u_expected, atol=1e-1)
    

def test_feedback_control():
    x = 0.0
    y = 0.0
    varphi = 0.0
    thetalist = np.array([0.,0.,0.2,-1.6,0.])
    state = np.hstack((np.array([varphi,x,y]).T,thetalist,np.array([0.,0.,0.,0.]).T))

    Tsb = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Tb0 = np.array([[1., 0., 0., 0.1662],[0., 1., 0., 0.],[0., 0., 1., 0.0026],[0., 0., 0., 1.]])
    M0e = np.array([
        [1., 0., 0., 0.033],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.6546],
        [0., 0., 0., 1.]])
    B = get_B()
    (_, T0e) = get_endeffector(state)
    X = np.matmul(np.matmul(Tsb,Tb0),T0e)

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
    Kp = np.identity(6)
    Ki = np.zeros((6,6))
    dt = 0.01

    (V, Xerr) = feedback_control(X, Xd, Xd_next, Kp, Ki, dt)
    (u, Je) = get_speeds(V, state)

    V_expected = np.array([0., 0.171, 0., 21.488, 0., 6.562])
    u_expected = np.array([157.5, 157.5, 157.5, 157.5, 0.0, -654.3, 1400.9, -746.8, 0.0])
    assert np.allclose(V, V_expected, atol=1e-3)
    assert np.allclose(u, u_expected, atol=1e-1)

def test_rotation():
    x = 0.0
    y = 0.0
    varphi = 0.0
    thetalist = np.array([0.,0.,0.2,-1.6,0.])
    state = np.hstack((np.array([varphi,x,y]).T,thetalist,np.array([0.,0.,0.,0.]).T))
    dt = 0.01

    varphi = 0.0
    X = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Xd = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    varphi = 1.234*dt
    Xd_next = np.array([[np.cos(varphi), -np.sin(varphi), 0., x],[np.sin(varphi), np.cos(varphi), 0., y],[0., 0., 1, 0.0963],[0., 0., 0., 1.]])
    Kp = np.zeros((6,6))
    Ki = np.zeros((6,6))

    (V, Xerr) = feedback_control(X, Xd, Xd_next, Kp, Ki, dt)
    (u, Je) = get_speeds(V, state)
    state = next_state(state, u, dt, 1e10)

def test_flatten():
    input_array = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[0,0,0,1]]])

    flat_array = flatten_output(input_array, 0.0)
    (output_array, gripper) = array_output(flat_array[0,:])
    assert(np.allclose(input_array, output_array, atol=1e-3))
    