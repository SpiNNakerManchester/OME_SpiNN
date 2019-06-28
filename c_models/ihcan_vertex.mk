APP = SpiNNakEar_IHCAN

SOURCES = ihcan_vertex/SpiNNakEar_IHCAN.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../model_binaries/)/

include $(SPINN_DIRS)/make/local.mk