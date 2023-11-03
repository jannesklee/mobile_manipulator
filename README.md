# Mobile Grasping Robot

This small program calculates the states for a mobile grapsing robot in order
to move a cube from one place to another. It

- calculates a trajectory,
- uses odometry,
- and calculates the states by feedback control.

The cube is thereby places from one to another position.

## Running the script
The script can be run with

    python mobman/mobman.py

It writes three outputs:

- *trajectory.csv* - the simulated trajectory
- *states.csv* - simulated states that follow the trajectory
- *Xerr.csv* - the error between the actual state and the wished trajectory

## Running the tests
You can run the tests by running

    python -m pytest -s

