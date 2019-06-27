APP = SpiNNakEar_DRNL

SOURCES = drnl_vertex/SpiNNakEar_DRNL.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../model_binaries/)/

include $(SPINN_DIRS)/make/local.mk