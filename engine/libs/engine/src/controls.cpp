#include <slm/engine/controls.hpp>

#include <imgui.h>

#include <cstddef>
#include <string>

namespace slm::controls {

void note(const std::string& text) {
    ImGui::TextWrapped("%s", text.c_str());
}

bool scalar(const std::string& label,
            float& value,
            const float minimum,
            const float maximum,
            const char* format) {
    ImGui::SetNextItemWidth(-1.0F);
    return ImGui::SliderFloat(
        label.c_str(), &value, minimum, maximum, format);
}

bool vector(const std::string& label,
            math::Vector& values,
            const float minimum,
            const float maximum,
            const char* format) {
    bool changed = false;
    ImGui::SeparatorText(label.c_str());
    ImGui::PushID(label.c_str());

    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::string item_label = "[" + std::to_string(index) + "]";
        ImGui::SetNextItemWidth(-1.0F);
        changed |= ImGui::SliderFloat(
            item_label.c_str(), &values[index], minimum, maximum, format);
    }

    ImGui::PopID();
    return changed;
}

bool matrix(const std::string& label,
            math::Matrix& values,
            const float minimum,
            const float maximum,
            const char* format) {
    bool changed = false;
    ImGui::SeparatorText(label.c_str());
    ImGui::PushID(label.c_str());

    for (std::size_t row = 0; row < values.rows(); ++row) {
        for (std::size_t column = 0; column < values.columns(); ++column) {
            const std::string item_label =
                "[" + std::to_string(row) + "," + std::to_string(column) + "]";
            ImGui::SetNextItemWidth(-1.0F);
            changed |= ImGui::SliderFloat(item_label.c_str(),
                                          &values(row, column),
                                          minimum,
                                          maximum,
                                          format);
        }
    }

    ImGui::PopID();
    return changed;
}

bool choice(const std::string& label,
            int& selected,
            const std::vector<std::string>& options) {
    if (options.empty()) {
        return false;
    }

    const bool valid_selection =
        selected >= 0 && selected < static_cast<int>(options.size());
    const char* preview = valid_selection
                              ? options[static_cast<std::size_t>(selected)].c_str()
                              : "Select";

    bool changed = false;
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::BeginCombo(label.c_str(), preview)) {
        for (std::size_t index = 0; index < options.size(); ++index) {
            const bool is_selected = selected == static_cast<int>(index);
            if (ImGui::Selectable(options[index].c_str(), is_selected)) {
                selected = static_cast<int>(index);
                changed = true;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool toggle(const std::string& label, bool& value) {
    return ImGui::Checkbox(label.c_str(), &value);
}

bool button(const std::string& label) {
    return ImGui::Button(label.c_str(), ImVec2(-1.0F, 0.0F));
}

}  // namespace slm::controls
