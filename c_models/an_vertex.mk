APP = SpiNNakEar_ANGroup

SOURCES = an_node/SpiNNakEar_AN_group_node.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../model_binaries/)/

include $(SPINN_DIRS)/make/local.mk