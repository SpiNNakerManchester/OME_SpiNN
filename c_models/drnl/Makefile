ifndef NEURAL_MODELLING_DIRS
    $(error NEURAL_MODELLING_DIRS is not set.  Please define NEURAL_MODELLING_DIRS (possibly by running "source setup" in the neural_modelling folder within the sPyNNaker source folder))
endif

APP := SpiNNakEar_DRNL
BUILD_DIR = build/
SOURCES = SpiNNakEar_DRNL.c
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))

CURRENT_DIR := $(dir $(MAKEFILE_PATH))
SOURCE_DIR := $(abspath $(CURRENT_DIR))/src
SOURCE_DIRS := $(SOURCE_DIR)

#CURRENT_DIR := $(dir $(MAKEFILE_PATH))
#EXTRA_SRC_DIR := $(abspath $(CURRENT_DIR)/src)
#SOURCE_DIRS += $(EXTRA_SRC_DIR)
#SOURCE_DIRS += $(SPINN_DIRS)/spinn_common/src
APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../OME_SpiNN/model_binaries/)/#$(abspath $(CURRENT_DIR)../model_binaries/)/
#CFLAGS +=-lm -I$(NEURAL_MODELLING_DIRS)/src -Isrc/ -Iinclude/ -I$(SPINN_DIRS)/spinn_common/include -ffast-math

LIBRARIES += -lm
CFLAGS +=-ffast-math -Iinclude/

#include $(NEURAL_MODELLING_DIRS)/src/Makefile.common
include $(SPINN_DIRS)/make/local.mk

