#pragma once

#include <slm/math/matrix.hpp>
#include <slm/math/vector.hpp>

#include <string>
#include <vector>

namespace slm::controls {

void note(const std::string& text);

bool scalar(const std::string& label,
            float& value,
            float minimum,
            float maximum,
            const char* format = "%.3f");

bool vector(const std::string& label,
            math::Vector& values,
            float minimum,
            float maximum,
            const char* format = "%.3f");

bool matrix(const std::string& label,
            math::Matrix& values,
            float minimum,
            float maximum,
            const char* format = "%.3f");

bool choice(const std::string& label,
            int& selected,
            const std::vector<std::string>& options);

bool toggle(const std::string& label, bool& value);
bool button(const std::string& label);

}  // namespace slm::controls
