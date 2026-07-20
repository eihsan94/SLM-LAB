#include <slm/engine/viewer.hpp>

#include <string>

int main(const int argument_count, char** arguments) {
    const std::string initial_lesson =
        argument_count > 1 ? arguments[1] : std::string{};
    return slm::engine::run_viewer(initial_lesson);
}
