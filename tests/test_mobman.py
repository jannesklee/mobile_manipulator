from mobman.mobman import next_state, feedback_control
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
])
def test_next_state(initial_state, speeds, expected):
    state = initial_state
    dt = 0.01

    for i in range(100):
        state = next_state(state, speeds, dt, 100)

    assert np.allclose(state[:3], expected[:3], atol=1e-3)

def test_feedback_control():
    X = np.array([
        [0.170 , 0.0 , 0.985 , 0.387],
        [0.0   , 1.0 , 0.0   , 0.0  ], 
        [-0.985, 0.0, 0.170, 0.570  ],
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
    dt = 0.1

    V = feedback_control(X, Xd, Xd_next, Kp, Ki, dt)

    expected = np.array([0., 0., 0., 20., 0., 10.])
    assert np.allclose(V, expected, atol=1e-3)

