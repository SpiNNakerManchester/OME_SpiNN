from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources import ConstantSDRAM
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource
from pacman.model.decorators.overrides import overrides
from pacman.executor.injection_decorator import inject_items

from data_specification.enums.data_type import DataType

from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.utilities import constants
from spinn_front_end_common.interface.simulation import simulation_utilities

from spinnak_ear.spinnak_ear_machine_vertices.abstract_ear_profiled import \
    AbstractEarProfiled

from enum import Enum
import numpy
import scipy.signal as sig


class OMEMachineVertex(
        MachineVertex, AbstractEarProfiled, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition):
    """ A vertex that runs the OME algorithm
    """

    # The number of bytes for the parameters
    _N_PARAMETER_BYTES = (9*4) + (6*8)

    # outgoing partition name from OME vertex
    OME_PARTITION_ID = "OMEData"

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    def __init__(
            self, data, fs, num_bfs, seq_size, time_scale=1, profile=True):
        """
        
        :param data: 
        :param fs: 
        :param num_bfs: 
        :param time_scale: 
        :param profile: 
        """

        MachineVertex.__init__(self, label="OME Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)

        self._data = data
        self._fs = fs
        self._num_bfs = num_bfs
        self._time_scale = time_scale
        self._seq_size = seq_size

        self._data_size = (
            (len(self._data) * DataType.FLOAT_64.size) + DataType.UINT32.size)

        # calculate stapes hpf coefficients
        wn = 1. / self._fs * 2. * 700.
        [self._shb, self._sha] = sig.butter(2, wn, 'high')

    @property
    def n_data_points(self):
        return len(self._data)

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        sdram += self._N_PARAMETER_BYTES + self._data_size
        sdram += self._num_bfs * DataType.UINT32.size
        sdram += self._profile_size()

        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=ConstantSDRAM(sdram),
            cpu_cycles=CPUCyclesPerTickResource(0),
            iptags=[], reverse_iptags=[])
        return resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "SpiNNakEar_OME.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return 1

    @inject_items({
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor"
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "routing_info", "tags", "placements", "machine_time_step",
            "time_scale_factor"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements,
            machine_time_step, time_scale_factor):

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve and write the parameters region
        region_size = self._N_PARAMETER_BYTES + self._data_size
        region_size += self._num_bfs * DataType.UINT32.size
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        self._reserve_profile_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step, time_scale_factor))

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the data size in words
        spec.write_value(len(self._data))

        # Write the CoreID
        spec.write_value(placement.p)

        # Write number of drnls
        spec.write_value(self._num_bfs)

        # Write the sampling frequency
        spec.write_value(self._fs)

        spec.write_value(self._num_bfs)

        # Write the key
        data_key = routing_info.get_first_key_from_pre_vertex(
            self, self.OME_PARTITION_ID)
        spec.write_value(data_key)

        # write the command key
        spec.write_value(0) # TODO: remove

        spec.write_value(self._time_scale)

        # write the stapes high pass filter coefficients
        # TODO:why is this needed?
        spec.write_value(0)

        # write the filter params
        for param in self._shb:
            spec.write_value(param, data_type=DataType.FLOAT_64)
        for param in self._sha:
            spec.write_value(param, data_type=DataType.FLOAT_64)

        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(self._data, dtype=numpy.double)
        spec.write_array(data.view(numpy.uint32))

        self._write_profile_dsg(spec)

        # End the specification
        spec.end_specification()
