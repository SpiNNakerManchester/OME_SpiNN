from pacman.model.graphs.machine import MachineVertex
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
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.constants import BYTES_PER_WORS
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.interface.profiling.profile_data \
    import ProfileData
from enum import Enum
import numpy

from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

from spinn_front_end_common.interface.profiling.abstract_has_profile_data \
    import AbstractHasProfileData
from spinn_front_end_common.interface.profiling import profile_utils
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.simulation import simulation_utilities
from data_specification.constants import APP_PTR_TABLE_BYTE_SIZE


class DRNLMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractHasProfileData,
        AbstractReceiveBuffersToHost):

    """ A vertex that runs the DRNL algorithm
    """

    # The number of bytes for the parameters
    # 1: n data points, 2: ome core, 3. my core, 4: ome app id?? 5: ack key?
    # 6: data key, 7: n ihcans 8: centre freq, 9: delay 10: sampling freq,
    # 11: ome data key, 12: recording flag, 13: n moc mvs, 14: conn lut size
    _N_PARAMETER_BYTES = 14 * BYTES_PER_WORS

    # The data type of each data element
    _DATA_ELEMENT_TYPE = DataType.FLOAT_32

    # The data type of the data count
    _DATA_COUNT_TYPE = DataType.UINT32

    # The data type of the keys
    _KEY_ELEMENT_TYPE = DataType.UINT32

    # the data type of the coreID
    _COREID_TYPE = DataType.UINT32

    _KEY_MASK_ENTRY_DTYPE = [
        ("key", "<u4"), ("mask", "<u4"),("conn_index","<u4")]
    _KEY_MASK_ENTRY_SIZE_BYTES = 12

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    MOC_RECORDING_REGION_ID = 0

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    def __init__(
            self, ome, cf, delay, is_recording=False, profile=True,
            drnl_index=None, data_partition_name="DRNLData",
            acknowledge_partition_name="DRNLDataAck"):
        """

        :param ome: The connected ome vertex
        """
        AbstractProvidesNKeysForPartition.__init__(self)
        MachineVertex.__init__(self, label="DRNL Node", constraints=None)
        self._ome = ome
        self._ome.register_processor(self)
        self._cf = cf
        self._fs = ome.fs
        self._delay = int(delay)
        self._mack = ome
        self._drnl_index = drnl_index
        self._is_recording = is_recording
        self._placement = None

        self._ihcan_vertices = list()
        self._ihcan_placements = list()
        self._mask = list()
        self._moc_vertices = list()

        self._num_data_points = ome.n_data_points
        self._n_moc_data_points = int(
            (self._num_data_points / (self._fs / 1000.0)) / 10) * 10
        self._recording_size = (
            self._n_moc_data_points * DataType.FLOAT_64.size +
            self._DATA_COUNT_TYPE.size)

        self._data_partition_name = data_partition_name
        self._acknowledge_partition_name = acknowledge_partition_name

        # Set up for profiling
        self._profile = profile
        self._n_profile_samples = 10000
        self._process_profile_times = None
        self._filter_params = self.calculate_filter_parameters()

    def register_processor(self, ihcan_vertex):
        self._ihcan_vertices.append(ihcan_vertex)

    def register_parent_processor(self, mack):
        self._mack=mack

    def get_acknowledge_key(self, placement, routing_info):
        key = routing_info.get_first_key_from_pre_vertex(
            placement.vertex, self._acknowledge_partition_name)
        return key

    def get_data_key(self,routing_info):
        key = routing_info.get_first_key_from_pre_vertex(
            self,self._data_partition_name)
        return key

    def add_moc_vertex(self,vertex,conn_matrix):
        self._moc_vertices.append((vertex,conn_matrix))

    def calculate_filter_parameters(self):
        dt = 1./self._fs
        nl_b_wq = 180.0
        nl_b_wp = 0.14
        nlin_bw = nl_b_wp * self._cf + nl_b_wq
        nlin_phi = 2.0 * numpy.pi * nlin_bw * dt
        nlin_theta = 2.0 * numpy.pi * self._cf * dt
        nlin_cos_theta = numpy.cos(nlin_theta)
        nlin_sin_theta = numpy.sin(nlin_theta)
        nlin_alpha = -numpy.exp(-nlin_phi) * nlin_cos_theta
        nlin_a1 = 2.0 * nlin_alpha
        nlin_a2 = numpy.exp(-2.0 * nlin_phi)
        nlin_z1 = complex(
            (1.0 + nlin_alpha * nlin_cos_theta), -
            (nlin_alpha * nlin_sin_theta))
        nlin_z2 = complex(
            (1.0 + nlin_a1 * nlin_cos_theta), -
            (nlin_a1 * nlin_sin_theta))
        nlin_z3 = complex(
            (nlin_a2 * numpy.cos(2.0 * nlin_theta)), -
            (nlin_a2 * numpy.sin(2.0 * nlin_theta)))
        nlin_tf = (nlin_z2 + nlin_z3) / nlin_z1
        nlin_b0 = abs(nlin_tf)
        nlin_b1 = nlin_alpha * nlin_b0

        lin_b_wq = 235.0
        lin_b_wp = 0.2
        lin_bw = lin_b_wp * self._cf + lin_b_wq
        lin_phi = 2.0 * numpy.pi * lin_bw * dt
        lin_c_fp = 0.62
        lin_c_fq = 266.0
        lin_cf = lin_c_fp * self._cf + lin_c_fq
        lin_theta = 2.0 * numpy.pi * lin_cf * dt
        lin_cos_theta = numpy.cos(lin_theta)
        lin_sin_theta = numpy.sin(lin_theta)
        lin_alpha = -numpy.exp(-lin_phi) * lin_cos_theta
        lin_a1 = 2.0 * lin_alpha
        lin_a2 = numpy.exp(-2.0 * lin_phi)
        lin_z1 = complex(
            (1.0 + lin_alpha * lin_cos_theta), -
            (lin_alpha * lin_sin_theta))
        lin_z2 = complex(
            (1.0 + lin_a1 * lin_cos_theta), -
            (lin_a1 * lin_sin_theta))
        lin_z3 = complex(
            (lin_a2 * numpy.cos(2.0 * lin_theta)), -
            (lin_a2 * numpy.sin(2.0 * lin_theta)))
        lin_tf = (lin_z2 + lin_z3) / lin_z1
        lin_b0 = abs(lin_tf)
        lin_b1 = lin_alpha * lin_b0

        return [lin_a1, lin_a2, lin_b0, lin_b1, nlin_a1, nlin_a2, nlin_b0,
                nlin_b1]

    @property
    def n_data_points(self):
        return self._num_data_points

    @property
    def fs(self):
        return self._fs

    @property
    def data_partition_name(self):
        return self._data_partition_name

    @property
    def acknowledge_partition_name(self):
        return self._acknowledge_partition_name

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        sdram = self._N_PARAMETER_BYTES
        sdram += constants.SYSTEM_BYTES_REQUIREMENT + 8
        sdram += len(self._moc_vertices) * self._KEY_MASK_ENTRY_SIZE_BYTES
        sdram += len(self._moc_vertices) * 256/4#max connlut size
        sdram += len(self._filter_params) * 8
        sdram += 4 * 8 * 4 #circular buffer to IHCs
        sdram += APP_PTR_TABLE_BYTE_SIZE


        if self._profile:
            sdram += profile_utils.get_profile_region_size(self._n_profile_samples)
        sdram += self._recording_size

        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=ConstantSDRAM(sdram),
            cpu_cycles=CPUCyclesPerTickResource(0),
            iptags=[], reverse_iptags=[])
        return resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "SpiNNakEar_DRNL.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return 4#2  # two for control IDs

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        if self._profile:
            profiles = profile_utils.get_profiling_data(
                1,
                self.PROFILE_TAG_LABELS, transceiver, placement)
            self._process_profile_times = profiles._tags['TIMER'][1]
        else:
            profiles=ProfileData(self.PROFILE_TAG_LABELS)
        return profiles

    @inject_items({
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
    })

    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "machine_time_step", "time_scale_factor", "routing_info",
            "tags", "placements"])
    def generate_data_specification(
            self, spec, placement,machine_time_step,
            time_scale_factor, routing_info, tags, placements):

        o_m_e_placement = placements.get_placement_of_vertex(self._ome).p
        self._placement = placements.get_placement_of_vertex(self)

        n_moc_mvs = len(self._moc_vertices)
        key_and_mask_table = numpy.zeros(n_moc_mvs, dtype=self._KEY_MASK_ENTRY_DTYPE)
        conn_lut = []
        conn_matrix_dict = {}
        for i,(moc,conn_matrix) in enumerate(self._moc_vertices):
            key_and_mask = routing_info.get_routing_info_from_pre_vertex(moc,'SPIKE').first_key_and_mask
            key_and_mask_table[i]['key']=key_and_mask.key
            key_and_mask_table[i]['mask']=key_and_mask.mask
            # key_and_mask_table[i]['conn_index']=int(len(conn_lut)/32)
            # for id in conn_matrix:
            #     conn_lut.append(id.item())
            conn_matrix_dict[str(key_and_mask.key)] = conn_matrix

        key_and_mask_table.sort(axis=0, order='key')
        for i,entry in enumerate(key_and_mask_table):
            conn_matrix = conn_matrix_dict[str(entry['key'])]
            key_and_mask_table[i]['conn_index'] = int(len(conn_lut) / 32)
            for id in conn_matrix:
                conn_lut.append(id.item())

        conn_lut_size = int(numpy.ceil(len(conn_lut)/32.))

        bitfield_conn_lut = numpy.zeros(conn_lut_size, dtype="uint32")
        for step, j in enumerate(range(0, len(conn_lut), 32)):
            lookup_bools = conn_lut[j:j + 32]
            indices = numpy.nonzero(lookup_bools)[0]
            for idx in indices:
                bitfield_conn_lut[step] |= 1 << (31 - idx)

        conn_lut_size_bytes = conn_lut_size*4
        key_and_mask_table_size_bytes = key_and_mask_table.size * self._KEY_MASK_ENTRY_SIZE_BYTES

        # Setup words + 1 for flags + 1 for recording size
        setup_size = constants.SYSTEM_BYTES_REQUIREMENT + 8
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=setup_size, label='systemInfo')

        # Reserve the parameters region
        region_size = self._N_PARAMETER_BYTES
        region_size += (1 + len(self._ihcan_vertices)) * self._KEY_ELEMENT_TYPE.size
        region_size += conn_lut_size_bytes + key_and_mask_table_size_bytes
        region_size += len(self._filter_params) * 8
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        #reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        if self._profile:
            #reserve profile region
            profile_utils.reserve_profile_region(
                spec, 1,
                self._n_profile_samples)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Get the placement of the vertices and find out how many chips
        # are needed
        keys = list()
        for vertex in self._ihcan_vertices:
            ihcan_placement = placements.get_placement_of_vertex(vertex)
            self._ihcan_placements.append(ihcan_placement)
            key = routing_info.get_first_key_from_pre_vertex(
                vertex, self._acknowledge_partition_name)
            keys.append(key)
        keys.sort()

        # Write the data size in words
        spec.write_value(
            self._num_data_points * (float(self._DATA_ELEMENT_TYPE.size) / 4.0),
            data_type=self._DATA_COUNT_TYPE)

        # Write the OMECoreID
        spec.write_value(
            o_m_e_placement, data_type=self._COREID_TYPE)

        # Write the CoreID
        spec.write_value(
            placement.p, data_type=self._COREID_TYPE)

        # Write the OMEAppID #TODO delete
        spec.write_value(
            0, data_type=self._COREID_TYPE)

        # Write the Acknowledge key #TODO delete
        spec.write_value(0)

        # Write the key
        if len(keys) > 0:
            data_key = routing_info.get_first_key_from_pre_vertex(
                self, self._data_partition_name)
            spec.write_value(data_key, data_type=DataType.UINT32)
        else:
            raise Exception("no drnl key generated!")

        # Write number of ihcans
        spec.write_value(
            len(self._ihcan_vertices), data_type=self._COREID_TYPE)

        # Write the centre frequency
        spec.write_value(self._cf, data_type=DataType.UINT32)

        # Write the delay
        spec.write_value(self._delay,data_type=DataType.UINT32)

        # Write the sampling frequency
        spec.write_value(self._fs,data_type=DataType.UINT32)

        # write the OME data key
        ome_data_key = routing_info.get_first_key_from_pre_vertex(
            self._ome, self._ome.data_partition_name)
        spec.write_value(ome_data_key, data_type=DataType.UINT32)

        # write is recording
        spec.write_value(int(self._is_recording), data_type=DataType.UINT32)

        # write the filter params
        for param in self._filter_params:
            spec.write_value(param, data_type=DataType.FLOAT_64)

        # Write the number of mocs
        spec.write_value(n_moc_mvs)
        # Write the size of the conn LUT
        spec.write_value(conn_lut_size)
        # Write the arrays
        spec.write_array(bitfield_conn_lut.view("uint32"))
        spec.write_array(key_and_mask_table.view("<u4"))

        if self._profile:
            profile_utils.write_profile_region_data(
                spec, 1,
                self._n_profile_samples)

        # Write the recording regions
        spec.switch_write_focus(self.REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # End the specification
        spec.end_specification()

    def read_moc_attenuation(self, buffer_manager, placement):
        """ Read back the spikes """

        data, _ = buffer_manager.get_data_by_placement(
            placement, self.MOC_RECORDING_REGION_ID)
        formatted_data = numpy.array(
            data, dtype=numpy.uint8, copy=True).view(numpy.float64)
        output_data = formatted_data.copy()
        output_length = len(output_data)

        # check all expected data has been recorded
        if output_length != self._n_moc_data_points:
            # if not set output to zeros of correct length, this will cause
            # an error flag in run_ear.py raise Warning
            print(
                "recording not complete, reduce Fs or disable RT!\n"
                "recorded output length:{}, expected length:{} "
                "at placement:{},{},{}".format(
                    len(output_data), self._n_moc_data_points, placement.x,
                    placement.y, placement.p))

            output_data.resize(self._n_moc_data_points, refcheck=False)
        return output_data

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        if self._is_recording:
            regions = [self.MOC_RECORDING_REGION_ID]
        else:
            regions = []
        return regions

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.RECORDING.value, txrx)