#include "model.h"

#include <iostream>

Model::Model(const struct ModelParameters &params) {
  gravity_vec_ = Vector3d(0, 0, -params.g);
  mass_ = params.mass;
  inertia_inv_ = params.inertia.inverse();
  motors_ = params.motors;

  Eigen::Quaterniond true_to_imu =
    Eigen::Quaterniond(Eigen::AngleAxisd(params.pitch_offset, Vector3d(0, 1, 0))) *
    Eigen::Quaterniond(Eigen::AngleAxisd(params.roll_offset, Vector3d(1, 0, 0)));
  imu_to_true_offset_ = true_to_imu.inverse();

  for (int i = 0; i < num_motors; i++) {
    Eigen::Quaterniond motor_q(
      params.motors[i].orientation(0),
      params.motors[i].orientation(1),
      params.motors[i].orientation(2),
      params.motors[i].orientation(3)
    );
    motor_directions_[i] = motor_q * Vector3d(1, 0, 0);
    motor_torque_directions_[i] = (params.motors[i].position - params.center_of_mass).cross(motor_directions_[i]);
  }
}

void Model::set_initial(const Vector3d &position, const Vector3d &velocity,
                        const Vector4d &orientation, const Vector3d &ang_vel) {
  position_ = position;
  velocity_ = velocity;

  Eigen::Quaterniond imu_q(
    orientation(0), orientation(1), orientation(2), orientation(3)
  );

  orientation_ = imu_to_true_offset_ * imu_q;
  ang_vel_ = ang_vel;
}

Vector6d Model::body_wrench(const Vector6d& rpms) {
  Vector6d wrench = Vector6d::Zero();
  for (int i = 0; i < num_motors; i++) {
    double force = motors_[i].c0 + motors_[i].c1 * rpms[i] + motors_[i].c2 * rpms[i] * rpms[i];
    double torque = motors_[i].cM * force;

    wrench.head(3) += force * motor_directions_[i];
    wrench.tail(3) += force * motor_torque_directions_[i];
    wrench.tail(3) += torque * motor_directions_[i];
  }
  return wrench;
}

void Model::step(double dt, const Vector3d &accel, const Vector3d &ang_accel) {
  position_ += velocity_ * dt + 0.5 * accel * dt * dt;
  velocity_ += accel * dt;

  Eigen::Matrix<double, 4, 1> new_quat = 0.5 * dt * (Eigen::Quaterniond(0.0, ang_vel_(0), ang_vel_(1), ang_vel_(2)) * orientation_).coeffs() + orientation_.coeffs();
  new_quat.normalize();
  orientation_.coeffs() = new_quat;
  ang_vel_ += ang_accel * dt;
}

std::pair<std::vector<Vector3d>, std::vector<Vector3d>> Model::predict(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &rpms, const std::vector<double> &rpm_times) {
  std::vector<Vector3d> velocities;
  std::vector<Vector3d> ang_vels;
  velocities.push_back(velocity_);
  ang_vels.push_back(ang_vel_);

  for (unsigned int i = 0; i < rpms.size(); i++) {
    Vector6d wrench = body_wrench(rpms[i]);
    wrench.head(3) /= mass_;
    wrench.tail(3) = (inertia_inv_ * wrench.tail(3)).eval();

    wrench.head(3) = (orientation_ * wrench.head(3)).eval();
    wrench.tail(3) = (orientation_ * wrench.tail(3)).eval();
    wrench.head(3) += gravity_vec_;

    step(rpm_times[i + 1] - rpm_times[i], wrench.head(3), wrench.tail(3));

    velocities.push_back(velocity_);
    ang_vels.push_back(ang_vel_);
  }
  return std::move(std::make_pair(velocities, ang_vels));
}
