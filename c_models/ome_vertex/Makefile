ifndef NEURAL_MODELLING_DIRS
    $(error NEURAL_MODELLING_DIRS is not set.  Please define NEURAL_MODELLING_DIRS (possibly by running "source setup" in the neural_modelling folder within the sPyNNaker source folder))
endif

APP := SpiNNakEar_OME
BUILD_DIR = build/
SOURCES = SpiNNakEar_OME.c
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(dir $(MAKEFILE_PATH))
SOURCE_DIR := $(abspath $(CURRENT_DIR))/src
SOURCE_DIRS := $(SOURCE_DIR)
#EXTRA_SRC_DIR := $(abspath $(CURRENT_DIR)/src)
#SOURCE_DIRS += $(EXTRA_SRC_DIR)
#SOURCE_DIRS += $(SPINN_DIRS)/spinn_common/src
APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../model_binaries/)/
CFLAGS +=-lm -Iinclude/ -ffast-math

include $(SPINN_DIRS)/make/local.mk
#include $(SPINN_DIRS)/make/app.make
#include $(SPINN_DIRS)/make/FrontEndCommon.mk
