import os
from collections import namedtuple
import numpy as np
from controllers import BaseController
import importlib
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator,  CONTROL_START_IDX, get_available_controllers, run_rollout
from matplotlib import pyplot as plt
import pandas as pd

data_path ='./step.csv'
model_path ='./models/tinyphysics.onnx'


State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

ACC_G = 9.81


if __name__ == "__main__":
	# test_cost, test_target_lataccel, test_current_lataccel, controller, roll_lataccel, v_ego, a_ego, action = run_rollout(data_path, "mycontroller", model_path, debug=True)
	tinyphysicsmodel = TinyPhysicsModel(model_path, debug=True)
	controller = importlib.import_module(f'controllers.{"mycontroller"}').Controller()

	df = pd.read_csv(data_path)
	processed_df = pd.DataFrame({
		'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
		'v_ego': df['vEgo'].values,
		'a_ego': df['aEgo'].values,
		'target_lataccel': df['targetLateralAcceleration'].values,
		'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
	})  
	# controller.kp = 0.1
	# controller.ki = 0.01
	# controller.kd = 0.0

	# sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=True)
	sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), processed_df, controller=controller, debug=True)
	test_cost=sim.rollout()
	test_target_lataccel = sim.target_lataccel_history
	test_current_lataccel = sim.current_lataccel_history 
	roll_lataccel = sim.data['roll_lataccel']
	v_ego = sim.data['v_ego']
	a_ego = sim.data['a_ego']
	action = sim.action_history



        # # Filter the current lateral acceleration
        # self.kf.predict()
        # self.kf.update(np.array([[current_lataccel]]))      
        # filtered_lataccel = self.kf.x[0, 0]  

        # # Kalman filter for lateral acceleration
        # self.kf = KalmanFilter(dim_x=1, dim_z=1)
        # self.kf.x = np.array([[0.]])  # Initial state estimate
        # self.kf.F = np.array([[1.]])  # State transition matrix
        # self.kf.H = np.array([[1.]])  # Measurement function
        # self.kf.P = np.array([[1.]])  # Initial covariance
        # self.kf.R = np.array([[0.5]])  # Measurement noise
        # self.kf.Q = np.array([[0.01]])  # Process noise        

