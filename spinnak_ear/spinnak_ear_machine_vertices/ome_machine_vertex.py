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
from spinn_front_end_common.abstract_models.\
    abstract_machine_supports_auto_pause_and_resume import \
    AbstractMachineSupportsAutoPauseAndResume
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.utilities.utility_objs import ExecutableType, \
    ProvenanceDataItem
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
        AbstractProvidesNKeysForPartition,
        AbstractMachineSupportsAutoPauseAndResume,
        ProvidesProvenanceDataFromMachineImpl):
    """ A vertex that runs the OME algorithm
    """

    __slots__ = [
        # input data
        "_data",
        # sampling freq
        "_fs",
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
        # timer period
        "_timer_period"
    ]

    # The number of bytes for the parameters
    # ints 1. total_ticks, 2. seq_size, 3. key, 4. timer_tick_period
    # floats 1. dt
    _N_PARAMETER_BYTES = (
        (4 * DataType.UINT32.size) + (1 * DataType.FLOAT_64.size))

    # 1. CONCHA_GAIN_SCALAR, 2. EAR_CANAL_GAIN_SCALAR
    _N_CONCHA_PARAMS_BYTES = 2 * DataType.FLOAT_64.size

    # The filter coeffs
    # 1. shb1, 2. shb2, 3. shb3, 4. sha1, 5.sha2, 6.sha3
    _N_FILTER_COEFFS_ITEMS = 6
    _N_FILTER_COEFFS_BYTES = _N_FILTER_COEFFS_ITEMS * DataType.FLOAT_64.size

    # outgoing partition name from OME vertex
    OME_PARTITION_ID = "OMEData"

    # ???????????????
    MAGIC_THREE = 10.0
    MAGIC_TWO = 700.0

    CONCHA_G = 0.25


    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('FILTER_COEFFS', 2),
               ('DATA', 3),
               ("CONCHA_PARAMS", 4),
               ('PROFILE', 5),
               ('PROVENANCE', 6)])

    # provenance items
    EXTRA_PROVENANCE_DATA_ENTRIES = Enum(
        value="EXTRA_PROVENANCE_DATA_ENTRIES",
        names=[("B0", 0),
               ("B1", 1),
               ("B2", 2),
               ("A0", 3),
               ("A1", 4),
               ("A2", 5),
               ("N_PROVENANCE_ELEMENTS", 6)])

    def __init__(
            self, data, fs, n_channels, seq_size, timer_period, profile=False):
        """ constructor for OME vertex

        :param data: the input data
        :param fs: the sampling freq
        :param n_channels: how many channels to process
        :param profile: bool stating if profiling or now
        """

        MachineVertex.__init__(self, label="OME Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)
        AbstractMachineSupportsAutoPauseAndResume.__init__(self)

        self._data = data
        self._fs = fs
        self._n_channels = n_channels
        self._seq_size = seq_size

        # size then list of doubles
        self._data_size = (
            (len(self._data) * DataType.FLOAT_64.size) + DataType.UINT32.size)

        # write timer period
        self._timer_period = timer_period

        # calculate stapes hpf coefficients
        wn = 1.0 / self._fs * 2.0 * self.MAGIC_TWO

        # noinspection PyTypeChecker
        [self._shb, self._sha] = sig.butter(2, wn, 'high')

    @overrides(AbstractMachineSupportsAutoPauseAndResume.my_local_time_period)
    def my_local_time_period(self, simulator_time_step):
        return self._timer_period

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.REGIONS.PROVENANCE.value

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        b0 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES.B0.value]
        b1 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES.B1.value]
        b2 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES.B2.value]
        a0 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES.A0.value]
        a1 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES. A1.value]
        a2 = provenance_data[self.EXTRA_PROVENANCE_DATA_ENTRIES.A2.value]

        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "b0"), b0))
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "b1"), b1))
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "b2"), b2))
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "a0"), a0))
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "a1"), a1))
        provenance_items.append(
            ProvenanceDataItem(self._add_name(names, "a2"), a2))
        return provenance_items

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
        # concha params
        sdram += self._N_CONCHA_PARAMS_BYTES
        # data
        sdram += self._data_size
        # filter coeffs
        sdram += self._N_FILTER_COEFFS_BYTES
        # profile
        sdram += self._profile_size()
        # provenance region
        sdram += self.get_provenance_data_size(
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value)

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

        # reserve concha params
        spec.reserve_memory_region(
            self.REGIONS.CONCHA_PARAMS.value, self._N_CONCHA_PARAMS_BYTES,
            "concha params")

        # reserve provenance data region
        self.reserve_provenance_data_region(spec)

        # reserve profiler region
        self._reserve_profile_memory_regions(spec)

    def _write_params(self, spec, routing_info):
        """ write the basic params region

        :param spec:  data spec
        :param routing_info: the keys holder
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

        # Write dt
        spec.write_value(1.0 / self._fs, DataType.FLOAT_64)

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

    def _write_concha_params(self, spec):
        spec.switch_write_focus(self.REGIONS.CONCHA_PARAMS.value)
        spec.write_value(pow(self.MAGIC_THREE, self.CONCHA_G))
        spec.write_value(pow(self.MAGIC_THREE, self.CONCHA_G))

    @inject_items({
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "local_time_step_map": "MachineTimeStepMap",
        "time_scale_factor": "TimeScaleFactor"
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "routing_info", "tags", "placements", "local_time_step_map",
            "time_scale_factor"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements,
            local_time_step_map, time_scale_factor):

        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), local_time_step_map[self],
            time_scale_factor))

        self._write_params(spec, routing_info)
        self._write_filter_coeffs(spec)
        self._write_input_data(spec)
        self._write_profile_dsg(spec)
        self._write_concha_params(spec)

        # End the specification
        spec.end_specification()
