from . import BaseController
import numpy as np
import math


class KalmanFilter:
    def __init__(self, initial_state, process_variance, measurement_variance):
        """
        Initialize the Kalman Filter.
        
        :param initial_state: Initial estimate of the state.
        :param process_variance: Process variance (how much the system can change).
        :param measurement_variance: Measurement variance (sensor noise).
        """
        self.state_estimate = initial_state
        self.estimation_error = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.kalman_gain = 0.0

    def update(self, measurement):
        """
        Update the state estimate with a new measurement.
        
        :param measurement: The new measurement value.
        :return: The updated state estimate.
        """
        # Prediction
        prediction = self.state_estimate
        prediction_error = self.estimation_error + self.process_variance
        
        # Update
        self.kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.state_estimate = prediction + self.kalman_gain * (measurement - prediction)
        self.estimation_error = (1 - self.kalman_gain) * prediction_error
        
        return self.state_estimate


class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self,):

        self.p =  0.44
        self.i =  self.p/5.1
        self.d = -self.p/2.9
        # print (self.p, self.i,self.d)
        self.error_integral_array = [0] * 10 * 60 *1
        self.prev_error = 0
        self.kf = KalmanFilter(initial_state=0.0, process_variance=1.05, measurement_variance=9)
        self.error_integral = 0
        self.prev_steering_action = 0




    def update(self, target_lataccel, current_lataccel, state, future_plan, step_idx, action):

        estimated_lataccel = self.kf.update(current_lataccel)
          
        error = (target_lataccel - estimated_lataccel)
        self.error_integral_array.pop(0)
        self.error_integral_array.append(error)
        error_integral_sum = sum(self.error_integral_array) 
        #self.error_integral += error
        self.error_integral = error_integral_sum
        error_diff = (error - self.prev_error) 
        self.prev_error = error
        steering_action= self.p * error + self.i* self.error_integral + self.d * error_diff 
        steering_action = 0.99 * steering_action + 0.01 * self.prev_steering_action
    
        # print(future_plan)
        self.prev_steering_action = steering_action
  
        return steering_action







# import os
# from collections import namedtuple
# import numpy as np
# from . import BaseController

# State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

# class KalmanFilter:
#     def __init__(self, initial_state, process_variance, measurement_variance):
#         self.state_estimate = initial_state
#         self.estimation_error = 1.0
#         self.process_variance = process_variance
#         self.measurement_variance = measurement_variance

#     def update(self, measurement):
#         prediction = self.state_estimate
#         prediction_error = self.estimation_error + self.process_variance
#         kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
#         self.state_estimate = prediction + kalman_gain * (measurement - prediction)
#         self.estimation_error = (1 - kalman_gain) * prediction_error
#         return self.state_estimate

# class SteeringController:
#     def __init__(self):

#         self.p =  0.44
#         self.i =  self.p/5.1
#         self.d = -self.p/2.9


#         self.error_integral = 0
#         self.prev_error = 0
#         self.prev_steering_action = 0


#         self.current_lataccel_filter = KalmanFilter(initial_state=0.0, process_variance=1.05, measurement_variance=9)



#     def update(self, current_lataccel, target_lataccel, state):
#         estimated_current_lataccel = self.current_lataccel_filter.update(current_lataccel)

#         error = target_lataccel - estimated_current_lataccel
#         self.error_integral += error
#         # self.error_integral = np.clip(self.error_integral, -self.max_integral, self.max_integral)

#         error_diff = error - self.prev_error


#         steering_action = (self.p * error) + (self.i * self.error_integral) + (self.d * error_diff)

#         steering_action = ( 0.99 * steering_action ) + (0.01 * self.prev_steering_action)

#         self.prev_error = error
#         self.prev_steering_action = steering_action

#         return steering_action

# class Controller(BaseController):
#     def __init__(self):
#         super().__init__()
#         self.controller = SteeringController()

#     def update(self, target_lataccel, current_lataccel, state, future_plan=None):
#         return self.controller.update(current_lataccel, target_lataccel, state)
