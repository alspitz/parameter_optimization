import numpy as np

from math_utils import quat_from_axis_angle
from model import Motor, HexModel

def acerodon01_all_c(c00, c01, c02, torque_const, i1, i2, i3, comx, comy, motor_constant, pitch_offset, roll_offset):
  import quad_model_py
  mp = quad_model_py.ModelParameters()
  mp.g = 9.80665
  mp.mass = 3.105
  mp.inertia = np.diag(np.array((i1, i2, i3)))
  mp.center_of_mass = np.array((comx, comy, 0.0))

  front_ang = 0.5236
  mid_ang = 1.5708
  back_ang = 2.618

  front_arm = 0.275
  mid_arm = 0.275
  back_arm = 0.275

  pos_1 = front_arm * np.array((np.cos(front_ang),  np.sin(front_ang), 0))
  pos_2 = mid_arm   * np.array((np.cos(mid_ang),    np.sin(mid_ang), 0))
  pos_3 = back_arm  * np.array((np.cos(back_ang),   np.sin(back_ang), 0))
  pos_4 = back_arm  * np.array((np.cos(back_ang),  -np.sin(back_ang), 0))
  pos_5 = mid_arm   * np.array((np.cos(mid_ang),   -np.sin(mid_ang), 0))
  pos_6 = front_arm * np.array((np.cos(front_ang), -np.sin(front_ang), 0))

  poss = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]

  for i in range(6):
    mp.motors[i].orientation = quat_from_axis_angle(np.array((0, 1, 0)), -np.pi / 2)
    mp.motors[i].position = poss[i]
    mp.motors[i].c0 = c00
    mp.motors[i].c1 = c01
    mp.motors[i].c2 = c02
    mp.motors[i].cM = [torque_const, -torque_const][i % 2]
    mp.motors[i].motor_constant = motor_constant

  mp.pitch_offset = pitch_offset
  mp.roll_offset = roll_offset

  return quad_model_py.Model(mp)

def hex_opt_for_all_c(c00, c01, c02, torque_const, i1, i2, comx, comy, motor_constant, pitch_offset, roll_offset):
  return acerodon01_all_c(c00, c01, c02, torque_const, i1, i2, 0.1, comx, comy, motor_constant, pitch_offset, roll_offset)

def hex_diff_motor(c1, c2, c3, c4, c5, c6, torque_const, i1, i2, comx, comy, motor_constant, pitch_offset, roll_offset):
  import quad_model_py
  mp = quad_model_py.ModelParameters()
  mp.g = 9.80665
  mp.mass = 3.105
  mp.inertia = np.diag(np.array((i1, i2, 0.1)))
  mp.center_of_mass = np.array((comx, comy, 0))

  front_ang = 0.5236
  mid_ang = 1.5708
  back_ang = 2.618

  front_arm = 0.275
  mid_arm = 0.275
  back_arm = 0.275

  cs = [c1, c2, c3, c4, c5, c6]

  pos_1 = front_arm * np.array((np.cos(front_ang),  np.sin(front_ang), 0))
  pos_2 = mid_arm   * np.array((np.cos(mid_ang),    np.sin(mid_ang), 0))
  pos_3 = back_arm  * np.array((np.cos(back_ang),   np.sin(back_ang), 0))
  pos_4 = back_arm  * np.array((np.cos(back_ang),  -np.sin(back_ang), 0))
  pos_5 = mid_arm   * np.array((np.cos(mid_ang),   -np.sin(mid_ang), 0))
  pos_6 = front_arm * np.array((np.cos(front_ang), -np.sin(front_ang), 0))

  poss = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]

  for i in range(6):
    mp.motors[i].orientation = quat_from_axis_angle(np.array((0, 1, 0)), -np.pi / 2)
    mp.motors[i].position = poss[i]
    mp.motors[i].c0 = 0
    mp.motors[i].c1 = 0
    mp.motors[i].c2 = cs[i]
    mp.motors[i].cM = [torque_const, -torque_const][i % 2]
    mp.motors[i].motor_constant = motor_constant

  mp.pitch_offset = pitch_offset
  mp.roll_offset = roll_offset

  return quad_model_py.Model(mp)

def acerodon01_model_c():
  return acerodon01_all_c(
    c00=0.035462,
    c01=-0.00015262,
    c02=1.6025e-7,
    torque_const=0.015,
    i1=0.04,
    i2=0.05,
    i3=0.07,
    comx=0.0,
    comy=0.0,
    motor_constant=10.35,
    pitch_offset=0.0,
    roll_offset=0.0
  )

def acerodon01_all(c00, c01, c02, torque_const, i1, i2, i3, motor_constant, pitch_offset, roll_offset):
  motor_thrust_model = lambda rpm: np.polynomial.polynomial.polyval(rpm, [c00, c01, c02])
  motor_torque_model_cw = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0.0, torque_const])
  motor_torque_model_ccw = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0.0, -torque_const])
  motor_quat = quat_from_axis_angle(np.array((0, 1, 0)), -np.pi / 2)

  front_ang = 0.5236
  mid_ang = 1.5708
  back_ang = 2.618

  front_arm = 0.275
  mid_arm = 0.275
  back_arm = 0.275

  pos_1 = front_arm * np.array((np.cos(front_ang),  np.sin(front_ang), 0))
  pos_2 = mid_arm   * np.array((np.cos(mid_ang),    np.sin(mid_ang), 0))
  pos_3 = back_arm  * np.array((np.cos(back_ang),   np.sin(back_ang), 0))
  pos_4 = back_arm  * np.array((np.cos(back_ang),  -np.sin(back_ang), 0))
  pos_5 = mid_arm   * np.array((np.cos(mid_ang),   -np.sin(mid_ang), 0))
  pos_6 = front_arm * np.array((np.cos(front_ang), -np.sin(front_ang), 0))
  torque_models = [motor_torque_model_cw, motor_torque_model_ccw]

  motors = [Motor(motor_thrust_model, torque_models[i % 2], pos, motor_quat) for i, pos in enumerate([pos_1, pos_2, pos_3, pos_4, pos_5, pos_6])]

  mass = 3.105
  inertia = np.diag(np.array((i1, i2, i3)))
  g = 9.80665

  return HexModel(g=g, mass=mass, inertia=inertia, motors=motors, motor_constant=motor_constant, pitch_offset=pitch_offset, roll_offset=roll_offset)

def hex_opt_for_all(c00, c01, c02, torque_const, i1, i2, motor_constant, pitch_offset, roll_offset):
  return acerodon01_all(c00, c01, c02, torque_const, i1, i2, 0.1, motor_constant, pitch_offset, roll_offset)

def hex_opt_for_motor(c00, c01, c02):
  return acerodon01_all(c00, c01, c02, 0.015, 0.04, 0.05, 0.07, 10.35, 0.0, 0.0)

def acerodon01_model():
  return acerodon01_all(
    c00=0.035462,
    c01=-0.00015262,
    c02=1.6025e-7,
    torque_const=0.015,
    i1=0.04,
    i2=0.05,
    i3=0.07,
    motor_constant=10.35,
    pitch_offset=0.0,
    roll_offset=0.0
  )

def danaus06_model():
  return danaus06_model_opt(8.65e-9, 0.0183, 0.003, 0.003, 0.005)

def danaus06_model_opt(ct, torque_const, i1, i2, i3):
  motor_thrust_model = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0, 0, ct])
  motor_torque_model_cw = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0.0, torque_const])
  motor_torque_model_ccw = lambda rpm: np.polynomial.polynomial.polyval(rpm, [0.0, -torque_const])
  motor_quat = quat_from_axis_angle(np.array((0, 1, 0)), -np.pi / 2)

  motor_ang = 0.925
  lever_arm = 0.1103

  pos_1 = lever_arm * np.array(( np.cos(motor_ang), -np.sin(motor_ang), 0.1))
  pos_2 = lever_arm * np.array((-np.cos(motor_ang),  np.sin(motor_ang), 0.1))
  pos_3 = lever_arm * np.array(( np.cos(motor_ang),  np.sin(motor_ang), 0.1))
  pos_4 = lever_arm * np.array((-np.cos(motor_ang), -np.sin(motor_ang), 0.1))
  torque_models = [motor_torque_model_cw, motor_torque_model_ccw]

  motors = [Motor(motor_thrust_model, torque_models[i < 2], pos, motor_quat) for i, pos in enumerate([pos_1, pos_2, pos_3, pos_4])]

  mass = 0.783
  inertia = np.diag(np.array((i1, i2, i3)))
  g = 9.80665

  return HexModel(g=g, mass=mass, inertia=inertia, motors=motors)
