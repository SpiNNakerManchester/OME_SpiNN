APP = SpiNNakEar_DRNL

SOURCE_DIRS := src/
SOURCE_DIRS += $(abspath $(NEURAL_MODELLING_DIRS)/src)
SOURCES = drnl_vertex/SpiNNakEar_DRNL.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../model_binaries/)/

include $(SPINN_DIRS)/make/local.mk
FEC_OPT = $(OSPACE)