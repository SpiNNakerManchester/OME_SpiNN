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

    __slots__ = [
        # input data
        "_data",
        # sampling freq
        "_fs" 
        # n channels in the ear
        "_n_channels",
        # seq size
        "_seq_size",
        # size of input data
        "_data_size",
        # filter coeffs
        "_shb",
        # filter coeffs
        "_sha",
    ]

    # The number of bytes for the parameters
    # ints 1. total_ticks, 2. seq_size, 3. key, 4. timer_tick_period
    # floats 1. dt
    _N_PARAMETER_BYTES = (
        (4 * DataType.UINT32.size) + (1 * DataType.FLOAT_64.size))

    # The filter coeffs
    # 1. shb1, 2. shb2, 3. shb3, 4. sha1, 5.sha2, 6.sha3
    _N_FILTER_COEFFS_BYTES = 6 * DataType.FLOAT_64.size

    # outgoing partition name from OME vertex
    OME_PARTITION_ID = "OMEData"

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('FILTER_COEFFS', 2),
               ('DATA', 3),
               ('PROFILE', 4)])

    def __init__(
            self, data, fs, n_channels, seq_size, profile=False):
        """ constructor for OME vertex
        
        :param data: the input data
        :param fs: the sampling freq
        :param n_channels: how many channels to process
        :param profile: bool stating if profiling or now
        """

        MachineVertex.__init__(self, label="OME Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)

        self._data = data
        self._fs = fs
        self._n_channels = n_channels
        self._seq_size = seq_size

        # size then list of doubles
        self._data_size = (
            (len(self._data) * DataType.FLOAT_64.size) + DataType.UINT32.size)

        # calculate stapes hpf coefficients
        wn = 1.0 / self._fs * 2.0 * 700.0

        # noinspection PyTypeChecker
        [self._shb, self._sha] = sig.butter(2, wn, 'high')

    @property
    def n_data_points(self):
        return len(self._data)

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        # system
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        # params
        sdram += self._N_PARAMETER_BYTES
        # data
        sdram += self._data_size
        # filter coeffs
        sdram += self._N_FILTER_COEFFS_BYTES
        # profile
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

    def _reserve_memory_regions(self, spec):
        """ reserve the dsg regions
        
        :param spec: data spec
        :rtype: None
        """

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve the parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value, self._N_PARAMETER_BYTES, "params")

        # reserve the filter coeffs
        spec.reserve_memory_region(
            self.REGIONS.FILTER_COEFFS.value, self._N_FILTER_COEFFS_BYTES,
            "filter")

        # reserve data region
        spec.reserve_memory_region(
            self.REGIONS.DATA.value, self._data_size, "data region")

        self._reserve_profile_memory_regions(spec)

    def _write_params(self, spec, routing_info, time_scale_factor):
        """ write the basic params region
        
        :param spec:  data spec
        :param routing_info: the keys holder
        :param time_scale_factor: the time scale factor
        :rtype: None 
        """

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write total ticks
        spec.write_value(len(self._data) / self._seq_size)

        # write seq size
        spec.write_value(self._seq_size)

        # Write the key
        data_key = routing_info.get_first_key_from_pre_vertex(
            self, self.OME_PARTITION_ID)
        spec.write_value(data_key)

        # write timer period
        spec.write_value((1e6 * self._seq_size / self._fs) * time_scale_factor)

        # Write dt
        spec.write_value(1.0 / self._fs, DataType.FLOAT_64)

        # write pi
        spec.write_value()

    def _write_filter_coeffs(self, spec):
        """ write filter coeffs to dsg
        
        :param spec: dsg writer
        :rtype: None 
        """

        spec.switch_write_focus(self.REGIONS.FILTER_COEFFS.value)

        # write the filter params
        for param in self._shb:
            spec.write_value(param, data_type=DataType.FLOAT_64)
        for param in self._sha:
            spec.write_value(param, data_type=DataType.FLOAT_64)

    def _write_input_data(self, spec):
        """ write input data to dsg
        
        :param spec: data spec writer
        :rtype: None 
        """

        spec.switch_write_focus(self.REGIONS.DATA.value)

        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(self._data, dtype=numpy.double)
        spec.write_array(data.view(numpy.uint32))

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

        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step, time_scale_factor))

        self._write_params(spec, routing_info, time_scale_factor)
        self._write_filter_coeffs(spec)
        self._write_input_data(spec)
        self._write_profile_dsg(spec)

        # End the specification
        spec.end_specification()
