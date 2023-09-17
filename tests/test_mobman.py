from mobman.mobman import next_state
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

    for i in range(3):
        assert state[i] == pytest.approx(expected[i], 1e-3)