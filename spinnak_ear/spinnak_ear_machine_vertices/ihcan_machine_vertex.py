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
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType

from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

from spinn_front_end_common.interface.profiling.abstract_has_profile_data \
    import AbstractHasProfileData
from spinn_front_end_common.interface.profiling import profile_utils
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.interface.profiling.profile_data \
    import ProfileData

from enum import Enum
import numpy

from data_specification.constants import APP_PTR_TABLE_BYTE_SIZE


class IHCANMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractHasProfileData,
        AbstractReceiveBuffersToHost
        ):
    """ A vertex that runs the DRNL algorithm
    """
    # The number of bytes for the parameters
    _N_PARAMETER_BYTES = 15*4#
    # The data type of each data element
    _DATA_ELEMENT_TYPE = DataType.FLOAT_32#DataType.FLOAT_64
    # The data type of the data count
    _DATA_COUNT_TYPE = DataType.UINT32
    # The numpy data type of each data element
    _NUMPY_DATA_ELEMENT_TYPE = numpy.single
    # The data type of the keys
    _KEY_ELEMENT_TYPE = DataType.UINT32
    # the data type of the coreID
    _COREID_TYPE = DataType.UINT32

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    def __init__(self, drnl,resample_factor,seed,is_recording,
                 n_fibres=2,ear_index=0,bitfield=True,profile=True,
                 n_lsr=0,n_msr=0,n_hsr=0):
        """
        :param ome: The connected ome vertex    """

        MachineVertex.__init__(self, label="IHCAN Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        self._is_recording = is_recording
        # self._minimum_buffer_sdram = minimum_buffer_sdram
        # self._buffered_sdram_per_timestep = buffered_sdram_per_timestep
        # self._spike_recorder = spike_recorder
        # self._buffer_size_before_receive = buffer_size_before_receive
        # self._max_spikes_per_ts = max_spikes_per_ts
        # self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._ear_index = ear_index

        self._drnl = drnl
        self._drnl.register_processor(self)
        self._resample_factor=resample_factor
        self._fs=drnl.fs
        self._n_atoms = n_fibres

        if n_lsr+n_msr+n_hsr > 2:
            raise Exception("Only 2 fibres can be modelled per IHCAN,"
                            " currently requesting {}lsr, {}msr, {}hsr".format(n_lsr,n_msr,n_hsr))
        self._n_lsr = n_lsr
        self._n_msr = n_msr
        self._n_hsr = n_hsr


        self._num_data_points = n_fibres * drnl.n_data_points # num of points is double previous calculations due to 2 fibre output of IHCAN model
        if bitfield:
            self._recording_size = numpy.ceil((self._num_data_points/8.) * self._KEY_ELEMENT_TYPE.size)
        else:
            self._recording_size = self._num_data_points * self._DATA_ELEMENT_TYPE.size
        self._seed = seed
        self._data_size = (
            self._num_data_points * self._DATA_ELEMENT_TYPE.size +
            self._DATA_COUNT_TYPE.size
        )
        self._sdram_usage = (
            self._N_PARAMETER_BYTES
        )
        # Set up for profiling
        self._n_profile_samples = 10000
        self._bitfield = bitfield
        self._profile = profile
        self._process_profile_times = None

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        sdram = self._N_PARAMETER_BYTES
        sdram += 1 * self._KEY_ELEMENT_TYPE.size
        sdram += constants.SYSTEM_BYTES_REQUIREMENT + 8
        sdram += self._recording_size
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
        return "SpiNNakEar_IHCAN.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return 4#for control IDs

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        if self._profile:
            profiles = profile_utils.get_profiling_data(
                self.REGIONS.PROFILE.value,
                self.PROFILE_TAG_LABELS, transceiver, placement)
            self._process_profile_times = profiles._tags['TIMER'][1]
        else:
            profiles=ProfileData(self.PROFILE_TAG_LABELS)
        return profiles

    @inject_items({
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "machine_graph":"MemoryMachineGraph",
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=["routing_info", "tags", "placements","machine_graph"
                              ,"machine_time_step","time_scale_factor"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements,machine_graph,
            machine_time_step,time_scale_factor):

        DRNL_placement=placements.get_placement_of_vertex(self._drnl).p

        # Setup words + 1 for flags + 1 for recording size
        setup_size = constants.SYSTEM_BYTES_REQUIREMENT + 8
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=setup_size, label='systemInfo')
        # Reserve and write the parameters region
        region_size = self._N_PARAMETER_BYTES
        region_size += 1 * self._KEY_ELEMENT_TYPE.size
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        #reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        if self._profile:
            #reserve profile region
            profile_utils.reserve_profile_region(
                spec, self.REGIONS.PROFILE.value,
                self._n_profile_samples)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        #write parameters
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the data size in words
        spec.write_value(
            self._num_data_points * (float(self._DATA_ELEMENT_TYPE.size) / 4.0),
            data_type=self._DATA_COUNT_TYPE)

        # Write the DRNLCoreID
        spec.write_value(
            DRNL_placement, data_type=self._COREID_TYPE)

        # Write the CoreID
        spec.write_value(
            placement.p, data_type=self._COREID_TYPE)

        #Write the DRNLAppID
        spec.write_value(
            0, data_type=self._COREID_TYPE)

        # Write the DRNL data key
        # spec.write_value(0)
        # spec.write_value(self._drnl.get_acknowledge_key(
        #     placement, routing_info))
        spec.write_value(self._drnl.get_data_key(routing_info))

        #Write the spike resample factor
        spec.write_value(
            self._resample_factor, data_type=self._COREID_TYPE)

        #Write the sampling frequency
        spec.write_value(
            self._fs, data_type=self._COREID_TYPE)

        #Write the routing key
        partitions = machine_graph \
            .get_outgoing_edge_partitions_starting_at_vertex(self)
        for partition in partitions:
            if partition.identifier == 'AN':
                rinfo = routing_info.get_routing_info_from_partition(
                    partition)
                key = rinfo.first_key
                mask = rinfo.first_mask
                spec.write_value(
                   key, data_type=self._COREID_TYPE)

        #Write is recording bool
        spec.write_value(int(self._is_recording),data_type=self._COREID_TYPE)

        #Write number of spontaneous fibres
        spec.write_value(int(self._n_lsr), data_type=self._COREID_TYPE)
        spec.write_value(int(self._n_msr), data_type=self._COREID_TYPE)
        spec.write_value(int(self._n_hsr), data_type=self._COREID_TYPE)

        #Write the seed
        data = numpy.array(self._seed, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

        # Write the recording regions
        spec.switch_write_focus(self.REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        #Write profile regions
        if self._profile:
            profile_utils.write_profile_region_data(
                spec, self.REGIONS.PROFILE.value,
                self._n_profile_samples)

        # End the specification
        spec.end_specification()

    def read_samples(self, buffer_manager, placement):
        """ Read back the spikes """

        # Read the data recorded
        # data_values, _ = buffer_manager.get_data_for_vertex(placement, 0)
        # data = data_values.read_all()
        data, _ = buffer_manager.get_data_by_placement(placement, 0)

        if self._bitfield:
            formatted_data = numpy.array(data, dtype=numpy.uint8)
            #only interested in every 4th byte!
            formatted_data = formatted_data[::4]
            lsr = formatted_data[0::2]#TODO:change names as output may not correspond to lsr + hsr fibres
            hsr = formatted_data[1::2]
            unpacked_lsr = numpy.unpackbits(lsr)
            unpacked_hsr = numpy.unpackbits(hsr)
            # output_data = [unpacked_lsr.astype(numpy.float32),unpacked_hsr.astype(numpy.float32)]
            output_data = numpy.asarray([numpy.nonzero(unpacked_lsr)[0]*(1000./self._fs),numpy.nonzero(unpacked_hsr)[0]*(1000./self._fs)])
            output_length = unpacked_hsr.size + unpacked_lsr.size

        else:
            numpy_format = list()
            numpy_format.append(("AN", numpy.float32))
            formatted_data = numpy.array(data, dtype=numpy.uint8,copy=True).view(numpy_format)
            output_data = formatted_data.copy()
            output_length = len(output_data)


        #check all expected data has been recorded
        if output_length != self._num_data_points:
            #if output not set to correct length it will cause an error flag in run_ear.py
            #raise Warning
            print("recording not complete, reduce Fs or disable RT!\n"
                            "recorded output length:{}, expected length:{} "
                            "at placement:{},{},{}".format(output_length,
                            self._num_data_points,placement.x,placement.y,placement.p))

            # output_data = numpy.zeros(self._num_data_points)
            output_data.resize(self._num_data_points,refcheck=False)
        #return formatted_data
        return output_data

    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        return recording_utilities.get_n_timesteps_in_buffer_space(
            buffer_space, 4)

    def get_recorded_region_ids(self):
        if self._is_recording:
            regions = [0]
        else:
            regions = []
        return regions

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.RECORDING.value, txrx)