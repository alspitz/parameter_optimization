import numpy as np

def parse_header(msg):
  return msg.header.stamp.to_sec()

def parse_vec(vec):
  return np.array((vec.x, vec.y, vec.z))

def parse_quat(quat):
  return np.array((quat.w, quat.x, quat.y, quat.z))

def parse_rpm(rpm_msg):
  return (np.array(rpm_msg.motor_rpm), parse_header(rpm_msg))

def parse_odom(odom_msg):
  class Odom:
    pass

  odom = Odom()
  odom.position = parse_vec(odom_msg.pose.pose.position)
  odom.velocity = parse_vec(odom_msg.twist.twist.linear)
  odom.quat = parse_quat(odom_msg.pose.pose.orientation)
  odom.ang_vel = parse_vec(odom_msg.twist.twist.angular)
  return (odom, parse_header(odom_msg))

def parse_event(event_msg):
  return event_msg.data
