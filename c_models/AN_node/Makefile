ifndef SPINN_DIRS
    $(error SPINN_DIRS is not set.  Please define SPINN_DIRS (possibly by running "source setup" in spinnaker_tools folder))
endif

# The name of the application to be built (binary will be this with a `.aplx`
# extension)
APP = SpiNNakEar_ANGroup

# Directory to place compilation artefacts (must include trailing slash)
BUILD_DIR = ./build/

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR))../../../OME_SpiNN/model_binaries/
#APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR))model_binaries/

# The sources to be compiled
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(dir $(MAKEFILE_PATH))
SOURCE_DIR := $(abspath $(CURRENT_DIR))/src
SOURCE_DIRS := $(SOURCE_DIR)

SOURCES = SpiNNakEar_AN_group_node.c
#LIBRARIES += -lm
CFLAGS +=-Iinclude/ -ffast-math

include $(SPINN_DIRS)/make/local.mk

all: $(APP_OUTPUT_DIR)$(APP).aplx
