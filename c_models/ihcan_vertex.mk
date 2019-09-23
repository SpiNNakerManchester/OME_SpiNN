APP = SpiNNakEar_IHCAN

SOURCE_DIRS := src/
SOURCE_DIRS += $(abspath $(NEURAL_MODELLING_DIRS)/src)

SOURCES = ihcan_vertex/SpiNNakEar_IHCAN.c \
          neuron/neuron_recording.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../spinnak_ear/model_binaries/)/

include $(SPINN_DIRS)/make/local.mk