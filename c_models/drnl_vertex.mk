# Copyright (c) 2019-2020 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

APP = SpiNNakEar_DRNL

SOURCE_DIRS := src/
SOURCE_DIRS += $(abspath $(NEURAL_MODELLING_DIRS)/src)
SOURCES = drnl_vertex/SpiNNakEar_DRNL.c \
          neuron/neuron_recording.c

CFLAGS += -DSPINNAKER

APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../spinnak_ear/model_binaries/)/

include $(SPINN_DIRS)/make/local.mk
FEC_OPT = $(OSPACE)