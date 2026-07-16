#include <slm/engine/viewer.hpp>

#include <slm/engine/lesson.hpp>
#include <slm/engine/show.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#define GL_SILENCE_DEPRECATION
#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#endif

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <implot3d.h>

namespace slm::engine {
namespace {

void glfw_error_callback(const int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](const char value) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(value)));
    });
    return text;
}

bool matches_search(const LessonEntry& lesson, const std::string& search) {
    if (search.empty()) {
        return true;
    }

    const std::string searchable =
        lowercase(lesson.title + " " + lesson.category + " " + lesson.id);
    return searchable.find(lowercase(search)) != std::string::npos;
}

void draw_sidebar(const std::vector<LessonEntry>& lessons,
                  int& selected_lesson,
                  char* search,
                  const std::size_t search_capacity) {
    ImGui::BeginChild("lesson_sidebar", ImVec2(280.0F, 0.0F), true);
    ImGui::TextColored(ImVec4(0.25F, 0.80F, 1.0F, 1.0F), "SLM Learning Lab");
    ImGui::TextDisabled("Build an SLM from scratch");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-1.0F);
    ImGui::InputTextWithHint(
        "##lesson_search", "Search lessons...", search, search_capacity);

    if (ImGui::Selectable("Dashboard", selected_lesson < 0)) {
        selected_lesson = -1;
    }
    ImGui::Separator();

    const std::string search_text = search;
    std::size_t category_begin = 0;
    while (category_begin < lessons.size()) {
        std::size_t category_end = category_begin + 1;
        while (category_end < lessons.size() &&
               lessons[category_end].category ==
                   lessons[category_begin].category) {
            ++category_end;
        }

        bool category_has_match = false;
        for (std::size_t index = category_begin; index < category_end; ++index) {
            category_has_match |= matches_search(lessons[index], search_text);
        }

        if (category_has_match) {
            if (!search_text.empty()) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Always);
            }
            const bool open = ImGui::CollapsingHeader(
                lessons[category_begin].category.c_str(),
                ImGuiTreeNodeFlags_DefaultOpen);
            if (open) {
                for (std::size_t index = category_begin; index < category_end;
                     ++index) {
                    if (!matches_search(lessons[index], search_text)) {
                        continue;
                    }
                    ImGui::PushID(lessons[index].id.c_str());
                    if (ImGui::Selectable(
                            lessons[index].title.c_str(),
                            selected_lesson == static_cast<int>(index))) {
                        selected_lesson = static_cast<int>(index);
                    }
                    ImGui::PopID();
                }
            }
        }
        category_begin = category_end;
    }

    ImGui::EndChild();
}

void draw_dashboard(const std::vector<LessonEntry>& lessons,
                    int& selected_lesson) {
    ImGui::TextColored(ImVec4(0.25F, 0.80F, 1.0F, 1.0F),
                       "Learning dashboard");
    ImGui::TextWrapped(
        "Choose a lesson from the syllabus. Lessons are automatically "
        "discovered from lessons/<topic>/lesson.cpp.");
    ImGui::Spacing();

    const float card_width = 220.0F;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const int cards_per_row = std::max(
        1, static_cast<int>((ImGui::GetContentRegionAvail().x + spacing) /
                            (card_width + spacing)));

    std::string current_category;
    int card_in_row = 0;
    for (std::size_t index = 0; index < lessons.size(); ++index) {
        const LessonEntry& lesson = lessons[index];
        if (lesson.category != current_category) {
            current_category = lesson.category;
            card_in_row = 0;
            ImGui::SeparatorText(current_category.c_str());
        }

        ImGui::PushID(lesson.id.c_str());
        if (ImGui::Button(lesson.title.c_str(), ImVec2(card_width, 72.0F))) {
            selected_lesson = static_cast<int>(index);
        }
        ImGui::PopID();

        ++card_in_row;
        const bool same_row = card_in_row < cards_per_row &&
                              index + 1 < lessons.size() &&
                              lessons[index + 1].category == current_category;
        if (same_row) {
            ImGui::SameLine();
        } else {
            card_in_row = 0;
        }
    }
}

void draw_lesson(const LessonEntry& lesson) {
    ImGui::TextColored(
        ImVec4(0.25F, 0.80F, 1.0F, 1.0F), "%s", lesson.title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%s", lesson.category.c_str());
    ImGui::Separator();

    show::detail::begin_frame();

    ImGui::BeginChild("lesson_controls", ImVec2(340.0F, 0.0F), true);
    ImGui::TextUnformatted("Experiment");
    ImGui::TextDisabled("These controls edit ordinary lesson variables.");
    if (lesson.draw != nullptr) {
        lesson.draw();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("lesson_views", ImVec2(0.0F, 0.0F), false);
    show::detail::render_views();
    ImGui::EndChild();
}

}  // namespace

int run_viewer(const std::string& initial_lesson_id) {
    std::vector<LessonEntry> lessons = registered_lessons();
    std::sort(lessons.begin(), lessons.end(), [](const LessonEntry& left,
                                                 const LessonEntry& right) {
        if (left.category != right.category) {
            return left.category < right.category;
        }
        if (left.order != right.order) {
            return left.order < right.order;
        }
        return left.title < right.title;
    });

    int selected_lesson = -1;
    if (!initial_lesson_id.empty()) {
        const auto selected =
            std::find_if(lessons.begin(),
                         lessons.end(),
                         [&initial_lesson_id](const LessonEntry& lesson) {
                             return lesson.id == initial_lesson_id;
                         });
        if (selected != lessons.end()) {
            selected_lesson =
                static_cast<int>(std::distance(lessons.begin(), selected));
        }
    }

    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE) {
        return 1;
    }

#if defined(__APPLE__)
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    GLFWwindow* window =
        glfwCreateWindow(1440, 860, "SLM Learning Lab", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGui::StyleColorsDark();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true) ||
        !ImGui_ImplOpenGL3_Init(glsl_version)) {
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    char search[128] = {};
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0F);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0F);
        ImGui::Begin("SLM Learning Lab Workspace", nullptr, flags);

        draw_sidebar(lessons, selected_lesson, search, sizeof(search));
        ImGui::SameLine();
        ImGui::BeginChild("main_workspace", ImVec2(0.0F, 0.0F), false);
        if (selected_lesson < 0) {
            draw_dashboard(lessons, selected_lesson);
        } else {
            draw_lesson(lessons[static_cast<std::size_t>(selected_lesson)]);
        }
        ImGui::EndChild();

        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::Render();

        int display_width = 0;
        int display_height = 0;
        glfwGetFramebufferSize(window, &display_width, &display_height);
        glViewport(0, 0, display_width, display_height);
        glClearColor(0.08F, 0.09F, 0.11F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}  // namespace slm::engine
