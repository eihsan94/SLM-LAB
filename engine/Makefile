.PHONY: all setup configure build run lesson clean

# Allow the memorable `make lesson my_topic` form. GNU Make normally treats
# `my_topic` as a second target, so it is captured as the lesson ID only when
# `lesson` is one of the requested goals.
ifneq ($(filter lesson,$(MAKECMDGOALS)),)
LESSON_POSITIONAL_GOALS := $(filter-out lesson,$(MAKECMDGOALS))
ifneq ($(word 2,$(LESSON_POSITIONAL_GOALS)),)
$(error Use one lesson ID: make lesson my_topic)
endif
LESSON_POSITIONAL_ID := $(firstword $(LESSON_POSITIONAL_GOALS))
ifneq ($(LESSON_POSITIONAL_ID),)
ifeq ($(origin NAME),command line)
$(error Use either `make lesson my_topic` or `make lesson NAME=my_topic`, not both)
endif
override NAME := $(LESSON_POSITIONAL_ID)
%:
	@:
endif
endif

CMAKE ?= cmake
BUILD_DIR ?= build
VCPKG_ROOT ?= $(CURDIR)/.local/vcpkg
VCPKG_REPOSITORY ?= https://github.com/microsoft/vcpkg.git
TOOLCHAIN := $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
CONFIGURE_STAMP := $(BUILD_DIR)/.slm-configured

all: build

setup:
	$(CMAKE) \
		-DVCPKG_ROOT="$(VCPKG_ROOT)" \
		-DVCPKG_REPOSITORY="$(VCPKG_REPOSITORY)" \
		-P cmake/bootstrap-vcpkg.cmake

$(CONFIGURE_STAMP): CMakeLists.txt CMakePresets.json vcpkg.json cmake/bootstrap-vcpkg.cmake | setup
	$(CMAKE) --preset dev \
		-B "$(BUILD_DIR)" \
		-DCMAKE_TOOLCHAIN_FILE="$(TOOLCHAIN)"
	$(CMAKE) -E touch "$(CONFIGURE_STAMP)"

configure: $(CONFIGURE_STAMP)

build: $(CONFIGURE_STAMP)
	$(CMAKE) --build "$(BUILD_DIR)" --config Debug
	@if [ -f "$(BUILD_DIR)/compile_commands.json" ]; then \
		$(CMAKE) -E copy_if_different \
			"$(BUILD_DIR)/compile_commands.json" compile_commands.json; \
	fi

run: build
	"$(BUILD_DIR)/bin/slm_lab" $(LESSON)

lesson:
	$(CMAKE) \
		-DLESSON_ID="$(NAME)" \
		-DLESSON_TITLE="$(TITLE)" \
		-DLESSON_CATEGORY="$(CATEGORY)" \
		-DLESSON_ORDER="$(ORDER)" \
		-DLESSON_TEMPLATE="$(TEMPLATE)" \
		-P cmake/new-lesson.cmake

clean:
	$(CMAKE) -E remove_directory "$(BUILD_DIR)"
	$(CMAKE) -E remove -f compile_commands.json
