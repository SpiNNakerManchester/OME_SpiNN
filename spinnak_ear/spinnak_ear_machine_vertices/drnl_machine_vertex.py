from pacman.model.graphs.abstract_sdram_partition import AbstractSDRAMPartition
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
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.constants import BYTES_PER_WORS
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.simulation import simulation_utilities


from spinnak_ear.spinnak_ear_machine_vertices.abstract_ear_profiled import \
    AbstractEarProfiled
from spinnak_ear.spinnak_ear_machine_vertices.ome_machine_vertex import \
    OMEMachineVertex

from enum import Enum
import numpy


class DRNLMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition, AbstractEarProfiled,
        AbstractReceiveBuffersToHost):

    """ A vertex that runs the DRNL algorithm
    """

    __slots__ = [
        "_cf",
        "_fs",
        "_delay",
        "_drnl_index",
        "_is_recording",
        "_moc_vertices",
        "_num_data_points",
        "_n_moc_data_points",
        "_recording_size",
        "_profile",
        "_n_profile_samples",
        "_process_profile_times",
        "_filter_params",
        "_seq_size"
    ]

    # the outgoing partition id for DRNL
    DRNL_PARTITION_ID = "DRNLData"
    DRNL_SDRAM_PARTITION_ID = "DRNLSDRAMData"

    # The number of bytes for the parameters
    # 1: n data points, 2: data key, 3: centre freq, 4: ome data key,
    # 5: recording flag, 6: seq size, 7:n buffers in sdram, 8: n mocs,
    # 9: size of mocs
    _N_PARAMS = 9
    _N_PARAMETER_BYTES = _N_PARAMS * BYTES_PER_WORS

    # circular buffer to IHCs
    N_BUFFERS_IN_SDRAM_TOTAL = 4

    # sdram edge address in sdram
    SDRAM_EDGE_ADDRESS_SIZE_IN_BYTES = 4

    # n filter params
    # 1. la1 2. la2 3. lb0 4. lb1 5. nla1 6. nla2 7.nlb0 8. nlb1
    N_FILTER_PARAMS = 8

    # n bytes for filter param region
    FILTER_PARAMS_IN_BYTES = (
        N_FILTER_PARAMS * constants.WORD_TO_BYTE_MULTIPLIER)

    # moc recording id
    MOC_RECORDING_REGION_ID = 0

    # regions
    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('FILTER_PARAMS', 2)
               ('RECORDING', 3),
               ('PROFILE', 4),
               ('SDRAM_EDGE_ADDRESS', 5)])

    def __init__(
            self, cf, delay, fs, n_data_points, drnl_index, is_recording,
            profile, seq_size):
        """

        :param ome: The connected ome vertex
        """
        MachineVertex.__init__(
            self, label="DRNL Node of {}".format(drnl_index), constraints=None)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)
        AbstractProvidesNKeysForPartition.__init__(self)

        self._cf = cf
        self._fs = fs
        self._delay = int(delay)
        self._drnl_index = drnl_index
        self._is_recording = is_recording
        self._seq_size = seq_size

        self._sdram_edge_size = (
            self.N_BUFFERS_IN_SDRAM_TOTAL * self._seq_size *
            DataType.FLOAT_64.size)

        self._moc_vertices = list()

        self._num_data_points = n_data_points
        self._n_moc_data_points = int(
            (self._num_data_points / (self._fs / 1000.0)) / 10) * 10

        # recording size
        self._recording_size = (
            self._n_moc_data_points * DataType.FLOAT_64.size +
            DataType.UINT32.size)

        # filter params
        self._filter_params = self._calculate_filter_parameters()

    @property
    def sdram_edge_size(self):
        return self._sdram_edge_size

    @property
    def drnl_index(self):
        return self._drnl_index

    def _get_data_key(self, routing_info):
        key = routing_info.get_first_key_from_pre_vertex(
            self, self.DRNL_PARTITION_ID)
        if key is None:
            raise Exception("no drnl key generated!")
        return key

    def _calculate_filter_parameters(self):
        """ magic maths for filter params. 
        
        :return: 
        """
        dt = 1.0 / self._fs
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
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        # system region
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        # sdram edge address store
        sdram += self.SDRAM_EDGE_ADDRESS_SIZE_IN_BYTES
        # the actual size needed by sdram edge
        sdram += self._sdram_edge_size
        # filter params
        sdram += self.FILTER_PARAMS_IN_BYTES
        # params
        sdram += self._N_PARAMETER_BYTES
        # profile
        sdram += self._profile_size()
        # reocrding
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
        return 1

    def _reserve_memory_regions(self, spec):
        """
        
        :param spec: 
        :return: 
        """

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve the parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value, self._N_PARAMETER_BYTES, "params")

        # reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1),
            "recording")

        spec.reserve_memory_region(
            self.REGIONS.SDRAM_EDGE_ADDRESS.value,
            self.SDRAM_EDGE_ADDRESS_SIZE_IN_BYTES, "sdram edge address")

        spec.reserve_memory_region(
            self.REGIONS.FILTER_PARAMS.value,
            self.FILTER_PARAMS_IN_BYTES, "filter params")

        # handle profile stuff
        self._reserve_profile_memory_regions(spec)

    def _write_param_region(self, spec, machine_graph, routing_info):
        """
        
        :param spec: 
        :param machine_graph: 
        :param routing_info: 
        :return: 
        """
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the data size in words
        spec.write_value(self._num_data_points)

        ome_data_key = None
        for edge in machine_graph.get_edges_ending_at_vertex(self):
            if isinstance(edge.pre_vertex, OMEMachineVertex):
               ome_data_key = routing_info.get_first_key_for_edge(edge)

        # Write the key
        spec.write_value(self._get_data_key(routing_info))

        # Write the sampling frequency
        spec.write_value(self._fs)

        # write the OME data key
        spec.write_value(ome_data_key)

        # write is recording
        spec.write_value(int(self._is_recording))

        # write seq size
        spec.write_value(self._seq_size)

        # write n buffers
        spec.write_value(self.N_BUFFERS_IN_SDRAM_TOTAL)

        # Write the number of mocs
        spec.write_value(0)
        # Write the size of the conn LUT
        spec.write_value(0)

    def _write_sdram_edge_rgion(self, spec, machine_graph):
        """
        
        :param spec: 
        :param machine_graph: 
        :return: 
        """
        for edge in machine_graph.get_edges_starting_at_vertex(self):
            partition = machine_graph.get_outgoing_partition_for_edge(edge)
            if isinstance(partition, AbstractSDRAMPartition):
                spec.write_value(partition.sdram_base_address)
                spec.write_value(partition.total_sdram_requirements)
                break

    def _write_filter_params(self, spec):
        """
        
        :param spec: 
        :return: 
        """
        spec.switch_write_focus(self.REGIONS.FILTER_PARAMS.value)
        for param in self._filter_params:
            spec.write_value(param, data_type=DataType.FLOAT_64)

    @inject_items({
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
        "machine_graph": "MemoryMachineGraph",
        "routing_info": "MemoryRoutingInfos",
        "placements": "MemoryPlacements",
        "tags": "MemoryTags",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "machine_time_step", "time_scale_factor", "machine_graph",
            "routing_info", "placements", "tags"])
    def generate_data_specification(
            self, spec, placement, machine_time_step, time_scale_factor,
            machine_graph, routing_info, placements, tags):

        # reserve regions
        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # params
        self._write_param_region(spec, machine_graph, routing_info)

        # write the filter params
        self._write_filter_params(spec)

        # sdram edge
        self._write_sdram_edge_rgion(spec, machine_graph)

        # only write params if used
        self._write_profile_dsg(spec)

        # Write the recording regions
        spec.switch_write_focus(self.REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # End the specification
        spec.end_specification()

    def read_moc_attenuation(self, buffer_manager, placement):
        """
         Read back the spikes 
        :param buffer_manager: buffer manager
        :param placement: placement
        :return: output data
        """

        data, _ = buffer_manager.get_data_by_placement(
            placement, self.MOC_RECORDING_REGION_ID)
        formatted_data = (
            numpy.array(data, dtype=numpy.uint8, copy=True).view(numpy.float64))
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
