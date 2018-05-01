#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;

typedef struct Motor {
  // thrust = c0 + c1 * RPM + c2 * RPM ^ 2
  double c0;
  double c1;
  double c2;
  // torque = cM * thrust(RPM)
  double cM;
  double motor_constant;
  Vector3d position;
  Vector4d orientation;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

} Motor;

typedef struct ModelParameters {
  double g;
  double mass;
  Matrix3d inertia;
  std::array<Motor, 6> motors;
  double pitch_offset; // 0 actual pitch = pitch_offset IMU pitch
  double roll_offset;
  Vector3d center_of_mass;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

} ModelParameters;

class Model {
  public:
    Model(const ModelParameters &params);
    void set_initial(const Vector3d &position, const Vector3d &velocity,
                     const Vector4d &orientation, const Vector3d &ang_vel);
    std::pair<std::vector<Vector3d>, std::vector<Vector3d>> predict(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &rpms, const std::vector<double> &rpm_times);

    static constexpr int num_motors = 6;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  private:
    Vector6d body_wrench(const Vector6d& rpms);
    void step(double dt, const Vector3d &accel, const Vector3d &ang_accel);

    Vector3d gravity_vec_;
    double mass_;
    Matrix3d inertia_inv_;
    Eigen::Quaterniond imu_to_true_offset_;

    Vector3d motor_directions_[num_motors];
    Vector3d motor_torque_directions_[num_motors];
    std::array<Motor, num_motors> motors_;

    Vector3d position_;
    Vector3d velocity_;
    Eigen::Quaterniond orientation_;
    Vector3d ang_vel_;
};
