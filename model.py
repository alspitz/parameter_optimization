from __future__ import division, print_function

import numpy as np

from math_utils import matrix_from_quat, quat_from_axis_angle, quat_inverse, quat_mult, quat_rotate
from rigid_body import RigidBody3D

def make_odom(rigid_body):
  class Odom:
    pass

  odom = Odom()
  odom.position = rigid_body.get_pos()
  odom.velocity = rigid_body.get_vel()
  odom.quat = rigid_body.get_quat()
  odom.ang_vel = quat_rotate(quat_inverse(odom.quat), rigid_body.get_ang())
  return odom

def integrate_rpms(desired_rpms, motor_constant, dt):
  true_rpms = np.empty(desired_rpms.shape)
  true_rpms[0] = desired_rpms[0]
  for i in range(1, len(desired_rpms)):
    true_rpms[i] = true_rpms[i - 1] + motor_constant * (desired_rpms[i] - true_rpms[i - 1]) * dt

  return true_rpms

class Motor:
  """
    thrust_f is from RPM to force
    torque_f is from force to torque
  """
  def __init__(self, thrust_f, torque_f, pos, quat):
    self.thrust_map = thrust_f
    self.torque_map = torque_f
    self.pos = pos
    self.quat = quat
    self.direction = quat_rotate(self.quat, np.array((1, 0, 0)))
    self.torque_direction = np.cross(self.pos, self.direction)

class HexModel:
  def __init__(self, **params):
    self.g = params['g']
    self.mass = params['mass']
    self.inertia = params['inertia']
    self.motors = params['motors']
    self.motor_constant = params['motor_constant']
    true_to_imu_offset = quat_mult(
      quat_from_axis_angle(np.array((0, 1, 0)), params['pitch_offset']),
      quat_from_axis_angle(np.array((1, 0, 0)), params['roll_offset'])
    )
    self.imu_to_true_offset = quat_inverse(true_to_imu_offset)

    self.world_gravity_wrench = np.array((0, 0, -self.g, 0, 0, 0))
    self.inertia_inv = np.linalg.pinv(self.inertia)

    self.thrust_map = np.empty((6, len(self.motors)))
    self.torque_map = np.empty((6, len(self.motors)))

    for i, motor in enumerate(self.motors):
      self.thrust_map[:3, i] = motor.direction
      self.thrust_map[3:, i] = motor.torque_direction
      self.torque_map[:3, i] = np.zeros(3)
      self.torque_map[3:, i] = motor.direction

    self.generalized_mass_inverse = np.zeros((6, 6))
    self.generalized_mass_inverse[:3, :3] = np.identity(3) / self.mass
    self.generalized_mass_inverse[3:, 3:] = self.inertia_inv

  def body_wrench(self, rpms):
    motor_thrusts = np.array([motor.thrust_map(rpm) for motor, rpm in zip(self.motors, rpms)])
    motor_torques = np.array([motor.torque_map(force) for motor, force in zip(self.motors, motor_thrusts)])

    total_wrench = self.thrust_map.dot(motor_thrusts) + \
                   self.torque_map.dot(motor_torques)
    return total_wrench

  def predict(self, rpms, rpm_times, **initial_args):
    """ rpm_times should have one more element than rpms """
    initial_args['quat'] = quat_mult(self.imu_to_true_offset, initial_args['quat'])
    # Need to rotate angular velocity into the world frame for RigidBody3D.
    initial_args['ang'] = quat_rotate(initial_args['quat'], initial_args['ang'])
    rigid_body = RigidBody3D(**initial_args)

    odoms = [make_odom(rigid_body)]

    for i, rpm_set in enumerate(rpms):
      body_wrench = self.body_wrench(rpm_set)
      body_accel_wrench = self.generalized_mass_inverse.dot(body_wrench)

      rot = matrix_from_quat(rigid_body.quat)
      world_accel_wrench = np.reshape(rot.dot(np.reshape(body_accel_wrench, (2, 3)).T).T, (6,))

      world_accel_wrench += self.world_gravity_wrench

      dt = rpm_times[i + 1] - rpm_times[i]
      rigid_body.step(dt, accel=world_accel_wrench[:3], ang_accel=world_accel_wrench[3:])

      odoms.append(make_odom(rigid_body))

    return odoms

if __name__ == "__main__":
  pos1 = np.array(( 0.3,  0,   0))
  pos2 = np.array((-0.3,  0,   0))
  pos3 = np.array(( 0.0,  0.3, 0))
  pos4 = np.array(( 0.0, -0.3, 0))
  pos5 = np.array(( 0.3,  0.3, 0))
  pos6 = np.array((-0.3, -0.3, 0))
  quat = quat_from_axis_angle(np.array((0, 1, 0)), -np.pi / 2)

  motor_thrust_model = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0, 0, 1e-9])
  motor_torque_model = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0, 0.015])
  motor_torque_model_neg = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0, -0.015])
  motor1 = Motor(motor_thrust_model, motor_torque_model, pos1, quat)
  motor2 = Motor(motor_thrust_model, motor_torque_model_neg, pos2, quat)
  motor3 = Motor(motor_thrust_model, motor_torque_model, pos3, quat)
  motor4 = Motor(motor_thrust_model, motor_torque_model_neg, pos4, quat)
  motor5 = Motor(motor_thrust_model, motor_torque_model, pos5, quat)
  motor6 = Motor(motor_thrust_model, motor_torque_model_neg, pos6, quat)

  inertia = np.diag(np.array((1, 2, 3)))

  hex_model = HexModel(
    g=9.81,
    mass=3.1,
    inertia=inertia,
    motors=[motor1, motor2, motor3, motor4, motor5, motor6]
  )

  print(hex_model.predict([[4500, 5500, 4500, 4500, 4500, 4500]]))
