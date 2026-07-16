#include <slm/engine/lesson.hpp>

namespace slm::engine {
namespace {

std::vector<LessonEntry>& mutable_lessons() {
    static std::vector<LessonEntry> lessons;
    return lessons;
}

}  // namespace

LessonRegistrar::LessonRegistrar(const char* id,
                                 const char* title,
                                 const char* category,
                                 const int order,
                                 const LessonDrawFunction draw) {
    mutable_lessons().push_back({id, title, category, order, draw});
}

const std::vector<LessonEntry>& registered_lessons() {
    return mutable_lessons();
}

}  // namespace slm::engine
