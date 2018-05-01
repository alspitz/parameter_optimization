#include "model.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(quad_model_py, m) {
  py::class_<Motor>(m, "Motor")
    .def(py::init<>())
    .def_readwrite("c0", &Motor::c0)
    .def_readwrite("c1", &Motor::c1)
    .def_readwrite("c2", &Motor::c2)
    .def_readwrite("cM", &Motor::cM)
    .def_readwrite("motor_constant", &Motor::motor_constant)
    .def_readwrite("position", &Motor::position)
    .def_readwrite("orientation", &Motor::orientation);

  py::class_<ModelParameters>(m, "ModelParameters")
    .def(py::init<>())
    .def_readwrite("g", &ModelParameters::g)
    .def_readwrite("mass", &ModelParameters::mass)
    .def_readwrite("inertia", &ModelParameters::inertia)
    .def_readwrite("motors", &ModelParameters::motors)
    .def_readwrite("pitch_offset", &ModelParameters::pitch_offset)
    .def_readwrite("roll_offset", &ModelParameters::roll_offset)
    .def_readwrite("center_of_mass", &ModelParameters::center_of_mass);

  py::class_<Model>(m, "Model")
    .def(py::init<const ModelParameters&>())
    .def("set_initial", &Model::set_initial)
    .def("predict", &Model::predict);
}
