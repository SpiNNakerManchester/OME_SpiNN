APP = SpiNNakEar_OME

SOURCES = ome_vertex/SpiNNakEar_OME.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../spinnak_ear/model_binaries/)/

include $(SPINN_DIRS)/make/local.mk
