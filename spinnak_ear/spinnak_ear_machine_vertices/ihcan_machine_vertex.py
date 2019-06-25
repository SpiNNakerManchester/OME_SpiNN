from pacman.model.graphs.common import EdgeTrafficType
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
from spinn_front_end_common.interface.buffer_management.buffer_models\
    .abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.interface.simulation import simulation_utilities

from spinnak_ear.spinnak_ear_machine_vertices.abstract_ear_profiled import \
    AbstractEarProfiled
from spinnak_ear.spinnak_ear_machine_vertices.drnl_machine_vertex import \
    DRNLMachineVertex

from enum import Enum
import numpy


class IHCANMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractReceiveBuffersToHost,
        AbstractEarProfiled):
    """ A vertex that runs the DRNL algorithm
    """

    # message when recording not complete
    RECORDING_WARNING = (
        "recording not complete, reduce Fs or disable RT!\n recorded output "
        "length:{}, expected length:{} at placement:{},{},{}")

    # The number of bytes for the parameters
    # 1. n data points, 2. drnl processor, 3. my core id, 4. drnl app id,
    # 5. drnl key, 6. resample factor, 7. fs, 8. my key, 9. is recording,
    # 10. n lsr, 11. n msr, 12. nhsr, 13,14,15,16 = seed
    _N_PARAMETERS = 16

    MAGIC_1 = 1000.0

    # The data type of each data element
    _DATA_ELEMENT_TYPE = DataType.FLOAT_32  #DataType.FLOAT_64

    # IHCAN partition id
    IHCAN_PARTITION_ID = "IHCANData"

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

    N_FIBRES_ERROR = (
        "Only {} fibres can be modelled per IHCAN, currently requesting {} "
        "lsr, {}msr, {}hsr")

    SPIKE_RECORDING_REGION_ID = 0

    def __init__(
            self, resample_factor, seed, is_recording, n_fibres, ear_index,
            bitfield, profile, fs, n_lsr, n_msr, n_hsr, max_n_fibres,
            drnl_data_points):
        """
        :param ome: The connected ome vertex    """

        MachineVertex.__init__(self, label="IHCAN Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)

        self._is_recording = is_recording
        self._ear_index = ear_index
        self._bitfield = bitfield

        self._re_sample_factor = resample_factor
        self._fs = fs
        self._n_atoms = n_fibres

        if n_lsr + n_msr + n_hsr > max_n_fibres:
            raise Exception(
                self.N_FIBRES_ERROR.format(max_n_fibres, n_lsr, n_msr, n_hsr))

        self._n_lsr = n_lsr
        self._n_msr = n_msr
        self._n_hsr = n_hsr

        # num of points is double previous calculations due to 2 fibre
        # output of IHCAN model
        self._num_data_points = n_fibres * drnl_data_points

        if self._bitfield:
            self._recording_size = numpy.ceil(
                (self._num_data_points / 8.) * DataType.UINT32.size)
        else:
            self._recording_size = (
                self._num_data_points * self._DATA_ELEMENT_TYPE.size)

        self._seed = seed

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        # params region
        sdram = self._N_PARAMETERS * constants.WORD_TO_BYTE_MULTIPLIER
        # system region
        sdram += constants.SYSTEM_BYTES_REQUIREMENT
        # recording region
        sdram += self._recording_size
        # profile region
        sdram += self._profile_size()

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

    @property
    def n_atoms(self):
        return self._n_atoms

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_atoms

    def _fill_in_parameter_region(
            self, spec, machine_graph, routing_info, placements, placement):

        drnl_processor = None
        drnl_key = None
        for edge in machine_graph.get_edges_ending_at_vertex(self):
            if isinstance(edge.pre_vertex, DRNLMachineVertex):
                drnl_processor = (
                    placements.get_placement_of_vertex(edge.pre_vertex).p)
                if edge.traffic_type == EdgeTrafficType.MULTICAST:
                    drnl_key = routing_info.get_first_key_for_edge(edge)

        # write parameters
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the data size in words
        spec.write_value(
            self._num_data_points *
            (float(self._DATA_ELEMENT_TYPE.size) /
             constants.WORD_TO_BYTE_MULTIPLIER))

        # Write the DRNLCoreID # TODO remove
        spec.write_value(drnl_processor)

        # Write the CoreID #TODO remove
        spec.write_value(placement.p)

        # Write the DRNLAppID # TODO remove
        spec.write_value(0)

        # Write the DRNL data key
        spec.write_value(drnl_key)

        # Write the spike resample factor
        spec.write_value(self._re_sample_factor)

        # Write the sampling frequency
        spec.write_value(self._fs)

        # Write the routing key
        partition = (
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(
                self)[0])
        r_info = routing_info.get_routing_info_from_partition(partition)
        key = r_info.first_key
        spec.write_value(key)

        # Write is recording bool
        spec.write_value(int(self._is_recording))

        # Write number of spontaneous fibres
        spec.write_value(int(self._n_lsr))
        spec.write_value(int(self._n_msr))
        spec.write_value(int(self._n_hsr))

        # Write the seed
        data = numpy.array(self._seed, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

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
        additional_arguments=[
            "routing_info", "tags", "placements", "machine_graph",
            "machine_time_step", "time_scale_factor"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements,
            machine_graph, machine_time_step, time_scale_factor):

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve and write the parameters region
        region_size = self._N_PARAMETERS * constants.WORD_TO_BYTE_MULTIPLIER
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        # reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        self._reserve_profile_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        self._fill_in_parameter_region(
            spec, machine_graph, routing_info, placements, placement)

        # Write the recording regions
        spec.switch_write_focus(self.REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # Write profile regions
        self._write_profile_dsg(spec)

        # End the specification
        spec.end_specification()

    def read_samples(self, buffer_manager, placement):
        """ Read back the spikes """

        # Read the data recorded
        # data_values, _ = buffer_manager.get_data_for_vertex(placement, 0)
        # data = data_values.read_all()
        data, _ = buffer_manager.get_data_by_placement(
            placement, self.SPIKE_RECORDING_REGION_ID)

        if self._bitfield:
            formatted_data = numpy.array(data, dtype=numpy.uint8)
            # only interested in every 4th byte!
            formatted_data = formatted_data[::4]
            # TODO:change names as output may not correspond to lsr + hsr fibres
            lsr = formatted_data[0::2]
            hsr = formatted_data[1::2]
            unpacked_lsr = numpy.unpackbits(lsr)
            unpacked_hsr = numpy.unpackbits(hsr)
            output_data = numpy.asarray(
                [numpy.nonzero(unpacked_lsr)[0] * (self.MAGIC_1 / self._fs),
                 numpy.nonzero(unpacked_hsr)[0] * (self.MAGIC_1 / self._fs)])
            output_length = unpacked_hsr.size + unpacked_lsr.size

        else:
            numpy_format = list()
            numpy_format.append(("AN", numpy.float32))
            formatted_data = numpy.array(
                data, dtype=numpy.uint8, copy=True).view(numpy_format)
            output_data = formatted_data.copy()
            output_length = len(output_data)

        # check all expected data has been recorded
        if output_length != self._num_data_points:
            # if output not set to correct length it will cause an error
            # flag in run_ear.pyraise Warning
            print(self.RECORDING_WARNING.format(
                output_length, self._num_data_points, placement.x, placement.y,
                placement.p))

            output_data.resize(self._num_data_points, refcheck=False)
        # return formatted_data
        return output_data

    def get_recorded_region_ids(self):
        if self._is_recording:
            regions = [0]
        else:
            regions = []
        return regions

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.RECORDING.value, txrx)
