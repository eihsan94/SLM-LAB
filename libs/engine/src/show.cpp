#include <slm/engine/show.hpp>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace slm::show {
namespace {

struct TextView {
    std::string title;
    std::string content;
};

struct MatrixView {
    std::string title;
    math::Matrix values;
    std::string description;
};

struct Plot2DView {
    std::string title;
    math::Vector x;
    math::Vector y;
    std::string series_label;
    AxisLabels2D labels;
    bool scatter = false;
};

struct Plot3DView {
    std::string title;
    math::Vector x;
    math::Vector y;
    math::Vector z;
    AxisLabels3D labels;
    bool connect_points = false;
};

using View = std::variant<TextView, MatrixView, Plot2DView, Plot3DView>;

std::vector<View>& frame_views() {
    static std::vector<View> views;
    return views;
}

const std::string& view_title(const View& view) {
    return std::visit([](const auto& item) -> const std::string& {
        return item.title;
    }, view);
}

void render(const TextView& view) {
    ImGui::TextWrapped("%s", view.content.c_str());
}

void render(const MatrixView& view) {
    if (!view.description.empty()) {
        ImGui::TextWrapped("%s", view.description.c_str());
        ImGui::Separator();
    }
    if (view.values.empty()) {
        ImGui::TextDisabled("The matrix is empty.");
        return;
    }

    const std::string table_id = view.title + "##table";
    const int column_count = static_cast<int>(view.values.columns() + 1);
    if (!ImGui::BeginTable(table_id.c_str(),
                           column_count,
                           ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                               ImGuiTableFlags_ScrollX |
                               ImGuiTableFlags_ScrollY)) {
        return;
    }

    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("row");
    for (std::size_t column = 0; column < view.values.columns(); ++column) {
        const std::string label = "c" + std::to_string(column);
        ImGui::TableSetupColumn(label.c_str());
    }
    ImGui::TableHeadersRow();

    for (std::size_t row = 0; row < view.values.rows(); ++row) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%zu", row);
        for (std::size_t column = 0; column < view.values.columns(); ++column) {
            ImGui::TableSetColumnIndex(static_cast<int>(column + 1));
            ImGui::Text("%.4f", view.values(row, column));
        }
    }
    ImGui::EndTable();
}

void render(const Plot2DView& view) {
    if (!ImPlot::BeginPlot(view.title.c_str(), ImVec2(-1.0F, -1.0F))) {
        return;
    }

    ImPlot::SetupAxes(view.labels.x.c_str(), view.labels.y.c_str());
    const std::size_t count = std::min(view.x.size(), view.y.size());
    if (count > 0) {
        if (view.scatter) {
            ImPlot::PlotScatter(view.series_label.c_str(),
                                view.x.data(),
                                view.y.data(),
                                static_cast<int>(count));
        } else {
            ImPlot::PlotLine(view.series_label.c_str(),
                             view.x.data(),
                             view.y.data(),
                             static_cast<int>(count));
        }
    }
    ImPlot::EndPlot();
}

void render(const Plot3DView& view) {
    if (!ImPlot3D::BeginPlot(view.title.c_str(), ImVec2(-1.0F, -1.0F))) {
        return;
    }

    ImPlot3D::SetupAxes(
        view.labels.x.c_str(), view.labels.y.c_str(), view.labels.z.c_str());
    const std::size_t count =
        std::min({view.x.size(), view.y.size(), view.z.size()});

    if (view.connect_points && count > 0) {
        ImPlot3D::PlotLine("Sample path",
                           view.x.data(),
                           view.y.data(),
                           view.z.data(),
                           static_cast<int>(count));
    }

    int hovered_point = -1;
    for (std::size_t index = 0; index < count; ++index) {
        const std::string point_id =
            "##" + view.title + "_point_" + std::to_string(index);
        ImPlot3D::PlotScatter(point_id.c_str(),
                              view.x.data() + index,
                              view.y.data() + index,
                              view.z.data() + index,
                              1);
        if (ImGui::IsItemHovered()) {
            hovered_point = static_cast<int>(index);
        }
    }

    if (hovered_point >= 0) {
        const std::size_t index = static_cast<std::size_t>(hovered_point);
        ImGui::BeginTooltip();
        ImGui::Text("Point %zu", index);
        ImGui::Separator();
        ImGui::Text("%s: %.4f", view.labels.x.c_str(), view.x[index]);
        ImGui::Text("%s: %.4f", view.labels.y.c_str(), view.y[index]);
        ImGui::Text("%s: %.4f", view.labels.z.c_str(), view.z[index]);
        ImGui::EndTooltip();
    }

    ImPlot3D::EndPlot();
}

}  // namespace

void text(const std::string& title, const std::string& content) {
    frame_views().emplace_back(TextView{title, content});
}

void matrix(const std::string& title,
            const math::Matrix& values,
            const std::string& description) {
    frame_views().emplace_back(MatrixView{title, values, description});
}

void plot2d(const std::string& title,
            const math::Vector& x,
            const math::Vector& y,
            const std::string& series_label,
            const AxisLabels2D& labels,
            const bool scatter) {
    frame_views().emplace_back(
        Plot2DView{title, x, y, series_label, labels, scatter});
}

void plot3d(const std::string& title,
            const math::Vector& x,
            const math::Vector& y,
            const math::Vector& z,
            const AxisLabels3D& labels,
            const bool connect_points) {
    frame_views().emplace_back(
        Plot3DView{title, x, y, z, labels, connect_points});
}

namespace detail {

void begin_frame() {
    frame_views().clear();
}

void render_views() {
    if (frame_views().empty()) {
        ImGui::TextDisabled("This lesson did not call any show::* helpers.");
        return;
    }

    if (!ImGui::BeginTabBar("lesson_visualizations")) {
        return;
    }

    for (const View& view : frame_views()) {
        if (ImGui::BeginTabItem(view_title(view).c_str())) {
            std::visit([](const auto& item) { render(item); }, view);
            ImGui::EndTabItem();
        }
    }
    ImGui::EndTabBar();
}

}  // namespace detail

}  // namespace slm::show
