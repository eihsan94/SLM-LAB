#pragma once

#include <slm/math/matrix.hpp>
#include <slm/math/vector.hpp>

#include <string>

namespace slm::show {

struct AxisLabels2D {
    std::string x = "X";
    std::string y = "Y";
};

struct AxisLabels3D {
    std::string x = "X";
    std::string y = "Y";
    std::string z = "Z";
};

void text(const std::string& title, const std::string& content);

void matrix(const std::string& title,
            const math::Matrix& values,
            const std::string& description = {});

void plot2d(const std::string& title,
            const math::Vector& x,
            const math::Vector& y,
            const std::string& series_label = "Series",
            const AxisLabels2D& labels = {},
            bool scatter = false);

void plot3d(const std::string& title,
            const math::Vector& x,
            const math::Vector& y,
            const math::Vector& z,
            const AxisLabels3D& labels = {},
            bool connect_points = false);

namespace detail {

void begin_frame();
void render_views();

}  // namespace detail

}  // namespace slm::show
