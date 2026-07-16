.PHONY: all setup configure build run lesson clean

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
