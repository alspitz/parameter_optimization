from __future__ import division, print_function

import sys
import time

import rosbag

import numpy as np

import bags
import model
import models

from msgs import parse_event, parse_odom, parse_rpm
from utils import plot3, plot_traj, scale_to_length_vecs

bag_meta = bags.BagAcerodonCircles
window_size = 100
rpm_skip = 100

class Bagfile:
  def __init__(self, bagfile, topic_map):
    self.topic_map = topic_map

    print("Loading bag...", end='')
    sys.stdout.flush()
    self.bag = rosbag.Bag(bagfile)
    self.messages = self.bag.read_messages(topics=[topic_map.rpm, topic_map.odom, topic_map.event])
    print("done.")

  def parse(self):
    print("Parsing bag...", end='')
    sys.stdout.flush()
    self.rpms = []
    self.odoms = []
    self.events = []
    for topic, msg, t in self.messages:
      if topic == self.topic_map.rpm:
        self.rpms.append(parse_rpm(msg))
      elif topic == self.topic_map.odom:
        self.odoms.append(parse_odom(msg))
      elif topic == self.topic_map.event:
        self.events.append((parse_event(msg), t.to_sec()))
      else:
        input("Error: Unrecognized topic:", topic)

    print("done.")

if __name__ == "__main__":
  bag = Bagfile(bag_meta.bagfile, bag_meta.topic_map)
  bag.parse()

  odoms, odom_times = zip(*bag.odoms)
  odoms = list(odoms)

  pos = np.array([odom.position for odom in odoms])
  desired_rpms = np.array([rpm for rpm, t in bag.rpms])
  rpm_times = [t for rpm, t in bag.rpms]

  start_time = None
  end_time = None

  for ev, t in bag.events:
    if ev == "TeleopEvent" and start_time is None:
      start_time = t
      print("Found first teleop at", start_time)
    elif ev == "HoverEvent" and start_time is None:
      start_time = t
      print("Found first hover at", start_time)
    elif ev == "TakeoffEvent" and start_time is None:
      start_time = t + 0.5
      print("Found first takeoff at", start_time)
    elif ev == "LandEvent" and end_time is None:
      end_time = t
      print("Found land event at", end_time)

  for i, (rpm, t) in enumerate(bag.rpms):
    if t > start_time:
      start_rpm_ind = i
      break

  data_pairs = []

  start_odom_ind = 0
  for i in range(start_rpm_ind, len(desired_rpms) - window_size - 1, rpm_skip):

    if end_time is not None and rpm_times[i] > end_time:
      break

    done = False

    # Find initial odometry
    while odom_times[start_odom_ind + 1] < rpm_times[i]:
      start_odom_ind += 1
      if start_odom_ind + 1 >= len(odom_times):
        done = True
        break

    if done:
      break

    rpm_set_times = rpm_times[i : i + window_size + 1]

    end_odom_ind = start_odom_ind
    while odom_times[end_odom_ind + 1] < rpm_set_times[-1]:
      end_odom_ind += 1
      if end_odom_ind + 1 >= len(odom_times):
        done = True
        break

    if done:
      break

    actual_odoms = odoms[start_odom_ind : end_odom_ind]
    #actual_pos = np.array([odom.position for odom in actual_odoms])
    actual_vel = np.array([odom.velocity for odom in actual_odoms])
    actual_ang = np.array([odom.ang_vel for odom in actual_odoms])

    #actual_pos_interp = scale_to_length_vecs(actual_pos, len(odoms_predicted))
    actual_vel_interp = scale_to_length_vecs(actual_vel, len(rpm_set_times))
    actual_ang_interp = scale_to_length_vecs(actual_ang, len(rpm_set_times))

    data_pairs.append((odoms[start_odom_ind], i, rpm_set_times, actual_vel_interp, actual_ang_interp))

  def cost(motor_constant, model_c):
    total_cost = 0

    rpms = model.integrate_rpms(desired_rpms, motor_constant, dt=0.005)

    for start_odom, rpm_start_ind, rpm_set_times, actual_vel_interp, actual_ang_interp in data_pairs:
      model_c.set_initial(start_odom.position, start_odom.velocity, start_odom.quat, start_odom.ang_vel)
      vels, ang_vels = model_c.predict(rpms[rpm_start_ind : rpm_start_ind + window_size], rpm_set_times)

      predict_vel = np.array(vels)
      predict_ang = np.array(ang_vels)
      #initial_odom = {
      #  'pos' : odoms[start_odom_ind].position,
      #  'vel' : odoms[start_odom_ind].velocity,
      #  'quat' : odoms[start_odom_ind].quat,
      #  'ang' : odoms[start_odom_ind].ang_vel
      #}

      #odoms_predicted = model.predict(rpm_sets, rpm_set_times, **initial_odom)
      ##predict_pos = np.array([odom.position for odom in odoms_predicted])
      #predict_vel = np.array([odom.velocity for odom in odoms_predicted])
      #predict_ang = np.array([odom.ang_vel for odom in odoms_predicted])

      #total_cost += np.mean(np.abs(actual_pos_interp - predict_pos))
      total_cost += np.mean(np.abs(actual_vel_interp - predict_vel)) + \
                    np.mean(np.abs(actual_ang_interp - predict_ang))

      #predict_ts = np.array(rpm_set_times) - rpm_set_times[0]
      #plot3((original_ts, predict_ts, predict_ts), (actual_pos, actual_pos_interp, predict_pos), ("original pos", "interpolated pos", "predicted pos"))
      #plot3((predict_ts, predict_ts), (actual_vel_interp, predict_vel), ("original vel", "interpolated vel", "predicted vel"))
      #plot3((predict_ts, predict_ts), (actual_ang_interp, predict_ang), ("original ang", "interpolated ang", "predicted ang"))

    return total_cost

  from scipy.optimize import differential_evolution

  bounds = [ (-0.1, 0.1), (-0.01, 0.01), (0, 1e-5), (0, 0.1), (0, 1), (0, 1), (-0.2, 0.2), (-0.2, 0.2), (1, 50), (-0.2, 0.2), (-0.2, 0.2) ]
  mc_ind = 8
  model_f_c = models.hex_opt_for_all_c
  #found_params = [ 2.44085810e-02, -3.98923431e-04, 1.79707497e-07, 3.32420690e-04, 5.96079739e-02, 4.95803241e-02, -3.05499454e-03, 8.00957750e-03, 1.79687614e+01, 2.79104002e-03, 1.92244996e-02]
  #print("Found cost is", cost(found_params[mc_ind], model_f_c(*found_params)))

  #bounds = [ (0, 1e-5), (0, 1e-5), (0, 1e-5), (0, 1e-5), (0, 1e-5), (0, 1e-5), (0, 0.1), (0, 1), (0, 1), (-0.2, 0.2), (-0.2, 0.2), (1, 50), (-0.2, 0.2), (-0.2, 0.2) ]
  #mc_ind=11
  #model_f_c = models.hex_diff_motor

  t = time.time()
  print("Original cost is", cost(10.35, models.acerodon01_model_c()))
  print(time.time() - t)

  cost_f = lambda x: cost(x[mc_ind], model_f_c(*x))

  def res_callback(xk, convergence):
    print("Params are", xk, "convergence is", convergence)

  result = differential_evolution(cost_f, bounds, maxiter=1000, disp=True, strategy='best1bin', popsize=5, callback=res_callback, polish=False)
  print(result.fun, result.x)
  print("Final cost is", cost_f(result.x))
  print("Window size was", window_size)
  print("rpm skip was", rpm_skip)
  print("Bag was", bag_meta.bagfile)
