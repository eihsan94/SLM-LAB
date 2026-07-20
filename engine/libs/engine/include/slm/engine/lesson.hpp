#pragma once

#include <string>
#include <vector>

namespace slm::engine {

using LessonDrawFunction = void (*)();

struct LessonEntry {
    std::string id;
    std::string title;
    std::string category;
    int order = 0;
    LessonDrawFunction draw = nullptr;
};

class LessonRegistrar {
public:
    LessonRegistrar(const char* id,
                    const char* title,
                    const char* category,
                    int order,
                    LessonDrawFunction draw);
};

const std::vector<LessonEntry>& registered_lessons();

}  // namespace slm::engine

// Defines and registers one frame function. Keep persistent lesson data in a
// normal `static` state object inside the function.
#define SLM_LESSON(id, title, category, order)                           \
    static void slm_lesson_draw_##id();                                  \
    namespace {                                                          \
    const ::slm::engine::LessonRegistrar slm_lesson_registrar_##id{      \
        #id, title, category, order, &slm_lesson_draw_##id};              \
    }                                                                    \
    static void slm_lesson_draw_##id()
