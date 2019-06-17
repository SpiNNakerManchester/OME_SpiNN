from pacman.model.graphs.machine import MachineVertex
from pacman.model.graphs.application import ApplicationVertex

from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
# from pacman.model.resources.sdram_resource import SDRAMResource
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
from spinn_front_end_common.interface.buffer_management.buffer_models\
    .abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.interface.profiling.profile_data \
    import ProfileData

from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition


from enum import Enum
import numpy
import scipy.signal as sig

from spinn_utilities.progress_bar import ProgressBar
from spinn_front_end_common.utilities import constants
from spinn_front_end_common.interface.profiling.abstract_has_profile_data \
    import AbstractHasProfileData
from spinn_front_end_common.interface.profiling import profile_utils
from spinn_front_end_common.interface.simulation import simulation_utilities
from data_specification.constants import APP_PTR_TABLE_BYTE_SIZE


class OMEMachineVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractHasProfileData):
    """ A vertex that runs the OME algorithm
    """

    # The number of bytes for the parameters
    _N_PARAMETER_BYTES = (9*4) + (6*8)
    # The data type of each data element
    _DATA_ELEMENT_TYPE = DataType.FLOAT_64#DataType.FLOAT_32#
    # The data type of the data count
    _DATA_COUNT_TYPE = DataType.UINT32
    # The numpy data type of each data element
    _NUMPY_DATA_ELEMENT_TYPE = numpy.double#numpy.single#
    # The data type of the keys
    _KEY_ELEMENT_TYPE = DataType.UINT32
    # the data type of the coreID
    _COREID_TYPE = DataType.UINT32

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    def __init__(self, data,fs,num_bfs,time_scale=1,profile=True,data_partition_name="OMEData",
            acknowledge_partition_name="OMEAck",command_partition_name="OMECommand"):
        """

        :param coordinator: The coordinator vertex
        :param model: The model being simulated
        """

        MachineVertex.__init__(self, label="OME Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        self._data = data
        self._data_partition_name = data_partition_name
        self._acknowledge_partition_name = acknowledge_partition_name
        self._command_partition_name = command_partition_name
        self._fs=fs
        self._num_bfs = num_bfs
        self._time_scale = time_scale

        self._drnl_vertices = list()
        self._drnl_placements = list()
        self._data_receiver = dict()
        self._mack_vertices = list()

        self._data_size = (
            (len(self._data) * self._DATA_ELEMENT_TYPE.size) +
            self._DATA_COUNT_TYPE.size
        )
        self._sdram_usage = (
            self._N_PARAMETER_BYTES + self._data_size
        )
        # calculate stapes hpf coefficients
        Wn = 1. / self._fs * 2. * 700.
        [self._shb, self._sha] = sig.butter(2, Wn, 'high')
        #matlab_output = loadmat('./hpf_coeffs.mat')
        #sha = matlab_output['stapesHighPass_a'].tolist()
        #self._sha = numpy.asarray(sha[0])
        #shb = matlab_output['stapesHighPass_b'].tolist()
        #self._shb = numpy.asarray(shb[0])

        # Set up for profiling
        self._profile = profile
        self._n_profile_samples = 10000
        self._process_profile_times = None

    @property
    def data_partition_name(self):
        return self._data_partition_name

    @property
    def acknowledge_partition_name(self):
        return self._acknowledge_partition_name

    @property
    def command_partition_name(self):
        return self._command_partition_name

    def register_processor(self, drnl_vertex):
        self._drnl_vertices.append(drnl_vertex)

    def register_mack_processor(self, mack_vertex):
        self._mack_vertices.append(mack_vertex)

    def get_acknowledge_key(self, placement, routing_info):
        key = routing_info.get_first_key_from_pre_vertex(
            placement.vertex, self._acknowledge_partition_name)
        return key

    def get_mask(self, placement, routing_info):
        mask = routing_info.get_routing_info_from_pre_vertex(
            self, self._data_partition_name).first_mask
        return ~mask & 0xFFFFFFFF

    @property
    def n_data_points(self):
        return len(self._data)
    @property
    def fs(self):
        return self._fs

    @property
    def get_num_bfs(self):
        return self._num_bfs

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        sdram = self._N_PARAMETER_BYTES + self._data_size
        sdram += len(self._drnl_vertices) * self._KEY_ELEMENT_TYPE.size
        sdram += constants.SYSTEM_BYTES_REQUIREMENT + 8
        sdram += APP_PTR_TABLE_BYTE_SIZE

        if self._profile:
            sdram += profile_utils.get_profile_region_size(self._n_profile_samples)

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
        # return ExecutableType.SYNC
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return 4 #for control IDs

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        if self._profile:
            profiles =  profile_utils.get_profiling_data(
                1,
                self.PROFILE_TAG_LABELS, transceiver, placement)
            self._process_profile_times = profiles._tags['TIMER'][1]
        else:
            profiles=ProfileData(self.PROFILE_TAG_LABELS)
        return profiles

    @inject_items({
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements"
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=["routing_info", "tags", "placements"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements):

        # Setup words + 1 for flags + 1 for recording size
        setup_size = constants.SYSTEM_BYTES_REQUIREMENT + 8
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=setup_size, label='systemInfo')

        # Reserve and write the parameters region
        region_size = self._N_PARAMETER_BYTES + self._data_size
        region_size += len(self._drnl_vertices) * self._KEY_ELEMENT_TYPE.size
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        if self._profile:
            #reserve profile region
            profile_utils.reserve_profile_region(
                spec, 1,
                self._n_profile_samples)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), 1,
            1))

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Get the placement of the vertices and find out how many chips
        # are needed
        keys = list()
        for vertex in self._drnl_vertices:
            drnl_placement = placements.get_placement_of_vertex(vertex)
            self._drnl_placements.append(drnl_placement)
            key = routing_info.get_first_key_from_pre_vertex(
                vertex, self._acknowledge_partition_name)
            keys.append(key)
        keys.sort()

        # Write the data size in words
        spec.write_value(
            len(self._data) ,
            data_type=self._DATA_COUNT_TYPE)

        # Write the CoreID
        spec.write_value(
            placement.p, data_type=self._COREID_TYPE)

        # Write number of drnls
        spec.write_value(
            len(self._drnl_vertices), data_type=self._COREID_TYPE)

        # Write number of macks
        spec.write_value(
            len(self._mack_vertices), data_type = self._COREID_TYPE)

        # Write the sampling frequency
        spec.write_value(
            self._fs, data_type=DataType.UINT32)

        spec.write_value(
            self._num_bfs, data_type=DataType.UINT32)

        # Write the key
        if len(keys)>0:
            data_key_orig = routing_info.get_routing_info_from_pre_vertex(
                self, self._data_partition_name).first_key
            data_key = routing_info.get_first_key_from_pre_vertex(
                self, self._data_partition_name)
            spec.write_value(data_key, data_type=DataType.UINT32)
        else:
            raise Exception("no ome key generated!")

        # write the command key
        # command_key = routing_info.get_first_key_from_pre_vertex(
        #     self,self._command_partition_name)
        # spec.write_value(command_key, data_type=DataType.UINT32)
        spec.write_value(0, data_type=DataType.UINT32) #TODO: remove

        spec.write_value(self._time_scale,data_type=DataType.UINT32)

        # write the stapes high pass filter coefficients
        spec.write_value(
            0, data_type=DataType.UINT32)#TODO:why is this needed?

        # write the filter params
        for param in self._shb:
            spec.write_value(param,data_type=DataType.FLOAT_64)
        for param in self._sha:
            spec.write_value(param,data_type=DataType.FLOAT_64)
        # spec.write_value(
        #     self._shb[0], data_type=self._DATA_ELEMENT_TYPE)
        # spec.write_value(
        #     self._shb[1], data_type=self._DATA_ELEMENT_TYPE)
        # spec.write_value(
        #     self._shb[2], data_type=self._DATA_ELEMENT_TYPE)
        # spec.write_value(
        #     self._sha[0], data_type=self._DATA_ELEMENT_TYPE)
        # spec.write_value(
        #     self._sha[1], data_type=self._DATA_ELEMENT_TYPE)
        # spec.write_value(
        #     self._sha[2], data_type=self._DATA_ELEMENT_TYPE)

        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(self._data, dtype=self._NUMPY_DATA_ELEMENT_TYPE)
        spec.write_array(data.view(numpy.uint32))

        if self._profile:
            profile_utils.write_profile_region_data(
                spec, self.REGIONS.PROFILE.value, self._n_profile_samples)

        # End the specification
        spec.end_specification()

    def read_samples(self, buffer_manager):
        """ Read back the samples
        """
        progress = ProgressBar(len(self._drnl_placements), "Reading results")
        samples = list()
        for placement in self._drnl_placements:

            # Read the data recorded
            samples.append(
                placement.vertex.read_samples(buffer_manager))
            progress.update()
        progress.end()

        # Merge all the arrays
        return numpy.hstack(samples)