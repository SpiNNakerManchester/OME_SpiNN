from __future__ import division
from pacman.model.graphs.abstract_sdram_partition import AbstractSDRAMPartition
from pacman.model.graphs.common import Slice
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ConstantSDRAM
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource
from pacman.model.decorators.overrides import overrides
from pacman.executor.injection_decorator import inject_items
from spinn_front_end_common.abstract_models import \
    AbstractSupportsBitFieldGeneration, \
    AbstractSupportsBitFieldRoutingCompression
from spinn_front_end_common.abstract_models.\
    abstract_machine_supports_auto_pause_and_resume import \
    AbstractMachineSupportsAutoPauseAndResume

from spinn_utilities.log import FormatAdapter

from data_specification.enums.data_type import DataType

from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.interface.buffer_management.buffer_models\
    .abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
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
import logging
import math

from spynnaker.pyNN.utilities import bit_field_utilities

logger = FormatAdapter(logging.getLogger(__name__))


class DRNLMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition, AbstractEarProfiled,
        AbstractReceiveBuffersToHost, AbstractSupportsBitFieldGeneration,
        AbstractSupportsBitFieldRoutingCompression,
        AbstractMachineSupportsAutoPauseAndResume):

    """ A vertex that runs the DRNL algorithm
    """

    __slots__ = [
        "_cf",
        "_fs",
        "_delay",
        "_drnl_index",
        "_num_data_points",
        "_n_moc_data_points",
        "_recording_size",
        "_profile",
        "_n_profile_samples",
        "_process_profile_times",
        "_filter_params",
        "_seq_size",
        "_synapse_manager",
        "_parent",
        "_n_buffers_in_sdram_total",
        "__on_chip_generatable_area",
        "__on_chip_generatable_size",
        "_neuron_recorder",
        "_timer_period"
    ]

    FAIL_TO_RECORD_MESSAGE = (
        "recording not complete, reduce Fs or disable RT!\n recorded output "
        "length:{}, expected length:{} at placement:{},{},{}")

    # the outgoing partition id for DRNL
    DRNL_PARTITION_ID = "DRNLData"
    DRNL_SDRAM_PARTITION_ID = "DRNLSDRAMData"

    # The number of bytes for the parameters
    #  1: data key, 2: ome data key,
    # 3: seq size, 4:n buffers in sdram, 5. n synapse types
    # 6. moc_resample_factor,
    _N_PARAMS = 6
    _N_PARAMETER_BYTES = _N_PARAMS * BYTES_PER_WORS

    # 1 moc_dec1, 2. moc_dec_2, 3 moc_dec_3, 4 moc_factor_1, 5 ctbm,
    # 6 recip_ctbm 7 disp_thresh
    _N_DOUBLE_PARAMS = 7
    _N_DOUBLE_PARAMS_BYTES = _N_DOUBLE_PARAMS * DataType.FLOAT_64.size

    # sdram edge address in sdram
    # 1. address, 2. size. 3. double elements
    SDRAM_EDGE_ADDRESS_SIZE_IN_WORDS = 3

    # n filter params
    # 1. la1 2. la2 3. lb0 4. lb1 5. nla1 6. nla2 7.nlb0 8. nlb1
    N_FILTER_PARAMS = 8

    # n bytes for filter param region
    FILTER_PARAMS_IN_BYTES = N_FILTER_PARAMS * DataType.FLOAT_64.size

    # moc recording id
    MOC_RECORDING_REGION_ID = 0

    # matrix weight scale
    GLOBAL_WEIGHT_SCALE = 1.0

    # synapse types
    N_SYNAPSE_TYPES = 2

    # moc magic numbers
    MOC_TAU_0 = 0.055
    MOC_TAU_1 = 0.4
    MOC_TAU_2 = 1
    MOC_TAU_WEIGHT = 0.9

    RATE_TO_ATTENTUATION_FACTOR = 6e2

    MOC_BUFFER_SIZE = 10

    MOC = "moc"

    # the params recordable from a drnl vertex
    RECORDABLES = [MOC]

    # recordable units NOTE RJ and ABS have no idea, but we're going with
    # meters for completeness on the MOC.
    RECORDABLE_UNITS = {MOC: 'meters'}

    # recording region id for the moc
    MOC_RECORDABLE_REGION_ID = 0

    # regions
    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('DOUBLE_PARAMS', 2),
               ('FILTER_PARAMS', 3),
               ('NEURON_RECORDING', 4),
               ('PROFILE', 5),
               ('SDRAM_EDGE_ADDRESS', 6),
               ('SYNAPSE_PARAMS', 7),
               ('POPULATION_TABLE', 8),
               ('SYNAPTIC_MATRIX', 9),
               ('SYNAPSE_DYNAMICS', 10),
               ('CONNECTOR_BUILDER', 11),
               ('DIRECT_MATRIX', 12),
               ('BIT_FIELD_FILTER', 13),
               ('BIT_FIELD_BUILDER', 14),
               ('BIT_FIELD_KEY_MAP', 15)])

    def __init__(
            self, cf, fs, n_data_points, drnl_index, profile, seq_size,
            synapse_manager, parent, n_buffers_in_sdram_total,
            neuron_recorder, timer_period):
        """ builder of the drnl machine vertex

        :param cf: ????????
        :param fs: sampling freequency of the OME
        :param n_data_points: the number of elements.....
        :param drnl_index:  the index in the list of drnls (used for slices)
        :param profile: bool flag saying if this vertex is set to profile
        :param seq_size: the size of a block
        :param synapse_manager: the synaptic manager
        :param parent: the app vertex
        :param n_buffers_in_sdram_total: the number of buffers in sequence in\
         the sdram edge
        :param neuron_recorder: the recorder for moc
        :param timer_period: the timer period of this core
        """

        MachineVertex.__init__(
            self, label="DRNL Node of {}".format(drnl_index), constraints=None)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)
        AbstractProvidesNKeysForPartition.__init__(self)

        # storage for the synapse manager locations
        self.__on_chip_generatable_offset = None
        self.__on_chip_generatable_size = None

        self._cf = cf
        self._fs = fs
        self._drnl_index = drnl_index
        self._seq_size = seq_size
        self._synapse_manager = synapse_manager
        self._parent = parent
        self._n_buffers_in_sdram_total = n_buffers_in_sdram_total
        self._neuron_recorder = neuron_recorder
        self._timer_period = timer_period

        self._sdram_edge_size = (
            self._n_buffers_in_sdram_total * self._seq_size *
            DataType.FLOAT_64.size)

        self._num_data_points = n_data_points

        # recording size
        self._recording_size_per_sim_time_step = (
            DataType.FLOAT_64.size * self.MOC_BUFFER_SIZE)

        # filter params
        self._filter_params = self._calculate_filter_parameters()

    @overrides(AbstractSupportsBitFieldRoutingCompression.
               key_to_atom_map_region_base_address)
    def key_to_atom_map_region_base_address(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=self.REGIONS.BIT_FIELD_KEY_MAP.value)

    @overrides(AbstractSupportsBitFieldGeneration.bit_field_builder_region)
    def bit_field_builder_region(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=self.REGIONS.BIT_FIELD_BUILDER.value)

    @overrides(AbstractSupportsBitFieldRoutingCompression.
               regeneratable_sdram_blocks_and_sizes)
    def regeneratable_sdram_blocks_and_sizes(self, transceiver, placement):
        base_address = \
            helpful_functions.locate_memory_region_for_placement(
                placement=placement, transceiver=transceiver,
                region=self.REGIONS.SYNAPTIC_MATRIX.value)
        return [(base_address + self.__on_chip_generatable_offset,
                 self.__on_chip_generatable_size)]

    @overrides(AbstractSupportsBitFieldGeneration.bit_field_base_address)
    def bit_field_base_address(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=self.REGIONS.BIT_FIELD_FILTER.value)

    @overrides(AbstractMachineSupportsAutoPauseAndResume.my_local_time_period)
    def my_local_time_period(self, simulator_time_step):
        return self._timer_period

    @staticmethod
    def get_matrix_scalar_data_types():
        return {DRNLMachineVertex.MOC: DataType.FLOAT_64}

    @staticmethod
    def get_matrix_output_data_types():
        return {DRNLMachineVertex.MOC: DataType.FLOAT_64}

    @property
    def sdram_edge_size(self):
        return self._sdram_edge_size

    @property
    def n_data_points(self):
        return self._num_data_points

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

        :return: list of 8 parameters used by the core.
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
    @inject_items({
        "graph": "MemoryApplicationGraph",
        "default_machine_time_step": "DefaultMachineTimeStep",
    })
    @overrides(
        MachineVertex.resources_required,
        additional_arguments={"graph", "default_machine_time_step"})
    def resources_required(self, graph, default_machine_time_step):
        # system region
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        # sdram edge address store
        sdram += (self.SDRAM_EDGE_ADDRESS_SIZE_IN_WORDS
                  * constants.WORD_TO_BYTE_MULTIPLIER)
        # bitfields bitfield region
        sdram += bit_field_utilities.get_estimated_sdram_for_bit_field_region(
            graph, self)
        # bitfield key map region
        sdram += bit_field_utilities.get_estimated_sdram_for_key_region(
            graph, self)
        # bitfield builder region
        sdram += bit_field_utilities.exact_sdram_for_bit_field_builder_region()
        # the actual size needed by sdram edge
        sdram += self._sdram_edge_size
        # filter params
        sdram += self.FILTER_PARAMS_IN_BYTES
        # params
        sdram += self._N_PARAMETER_BYTES
        # double params
        sdram += self._N_DOUBLE_PARAMS_BYTES
        # profile
        sdram += self._profile_size()
        # synapses
        sdram += self._synapse_manager.get_sdram_usage_in_bytes(
            Slice(self._drnl_index, self._drnl_index + 1),
            graph.get_edges_ending_at_vertex(self._parent),
            default_machine_time_step)
        # recording stuff
        sdram += self._neuron_recorder.get_sdram_usage_in_bytes(
            Slice(self._drnl_index, self._drnl_index))
        variable_sdram = self._neuron_recorder.get_variable_sdram_usage(
            Slice(self._drnl_index, self._drnl_index))

        # find variable sdram
        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=variable_sdram + ConstantSDRAM(sdram),
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

    def _reserve_memory_regions(
            self, spec, machine_graph, n_key_map, vertex):
        """ reserve memory regions

        :param spec: spec
        :param machine_graph: machine graph
        :param n_key_map: map between partitions and n keys
        :param vertex: machine vertex
        :rtype: None
        """

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve the parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value, self._N_PARAMETER_BYTES, "params")

        # Reserve the double params region
        spec.reserve_memory_region(
            self.REGIONS.DOUBLE_PARAMS.value, self._N_DOUBLE_PARAMS_BYTES,
            "double params")

        # reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.NEURON_RECORDING.value,
            self._neuron_recorder.get_static_sdram_usage(
                Slice(self._drnl_index, self._drnl_index)),
            "recording")

        # sdram edge addresses
        spec.reserve_memory_region(
            self.REGIONS.SDRAM_EDGE_ADDRESS.value,
            (self.SDRAM_EDGE_ADDRESS_SIZE_IN_WORDS *
             constants.WORD_TO_BYTE_MULTIPLIER), "sdram edge address")

        # filter param regions
        spec.reserve_memory_region(
            self.REGIONS.FILTER_PARAMS.value,
            self.FILTER_PARAMS_IN_BYTES, "filter params")

        # bitfields region
        bit_field_utilities.reserve_bit_field_regions(
            spec, machine_graph, n_key_map, vertex,
            self.REGIONS.BIT_FIELD_BUILDER.value,
            self.REGIONS.BIT_FIELD_FILTER.value,
            self.REGIONS.BIT_FIELD_KEY_MAP.value)

        # handle profile stuff
        self._reserve_profile_memory_regions(spec)

    def _write_param_region(self, spec, machine_graph, routing_info):
        """ writes the param region

        :param spec: spec
        :param machine_graph: machine graph
        :param routing_info: the holder of keys
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        ome_data_key = None
        for edge in machine_graph.get_edges_ending_at_vertex(self):
            if isinstance(edge.pre_vertex, OMEMachineVertex):
                ome_data_key = routing_info.get_first_key_for_edge(edge)

        # Write the key
        spec.write_value(self._get_data_key(routing_info))

        # write the OME data key
        spec.write_value(ome_data_key)

        # write seq size
        spec.write_value(self._seq_size)

        # write n buffers
        spec.write_value(self._n_buffers_in_sdram_total)

        # write n synapses
        spec.write_value(self.N_SYNAPSE_TYPES)

        # write moc resample factor
        spec.write_value(self._fs / 1000.0)

    def _write_double_params_region(self, spec):
        """ writes the parameters which are double types

        :param spec: data spec writer
        :rtype: None
        """

        spec.switch_write_focus(self.REGIONS.DOUBLE_PARAMS.value)

        # moc dec 1
        dt = 1.0 / self._fs
        spec.write_value(
            math.exp(-dt / self.MOC_TAU_0), data_type=DataType.FLOAT_64)

        # moc dec 2
        spec.write_value(
            math.exp(-dt / self.MOC_TAU_1), data_type=DataType.FLOAT_64)

        # moc dec 3
        spec.write_value(
            math.exp(-dt / self.MOC_TAU_2), data_type=DataType.FLOAT_64)

        # moc_factor_1
        spec.write_value(
            self.RATE_TO_ATTENTUATION_FACTOR * self.MOC_TAU_WEIGHT * dt,
            data_type=DataType.FLOAT_64)

        # ctbm
        ctbm = 1e-9 * math.pow(10.0, 32.0 / 20.0)
        spec.write_value(ctbm, data_type=DataType.FLOAT_64)

        # recip_ctbm
        spec.write_value(1.0 / ctbm, data_type=DataType.FLOAT_64)

        # disp_thresh
        spec.write_value(ctbm / 30e4, data_type=DataType.FLOAT_64)

    def _write_sdram_edge_region(self, spec, machine_graph):
        """ writes data for the sdram edge reading

        :param spec: the data spec writer
        :param machine_graph: the machine graph
        :rtype: None
        """

        spec.switch_write_focus(self.REGIONS.SDRAM_EDGE_ADDRESS.value)
        for edge in machine_graph.get_edges_starting_at_vertex(self):
            partition = machine_graph.get_outgoing_partition_for_edge(edge)
            if (isinstance(partition, AbstractSDRAMPartition) and
                    partition.identifier == self.DRNL_SDRAM_PARTITION_ID):
                spec.write_value(partition.sdram_base_address)
                spec.write_value(partition.total_sdram_requirements())
                spec.write_value(
                    partition.total_sdram_requirements() /
                    DataType.FLOAT_64.size)
                break

    def _write_filter_params(self, spec):
        """ writes the filter params

        :param spec: specification writer
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.FILTER_PARAMS.value)
        for param in self._filter_params:
            spec.write_value(param, data_type=DataType.FLOAT_64)

    @inject_items({
        "time_period_map": "MachineTimeStepMap",
        "time_scale_factor": "TimeScaleFactor",
        "machine_graph": "MemoryMachineGraph",
        "application_graph": "MemoryApplicationGraph",
        "routing_info": "MemoryRoutingInfos",
        "placements": "MemoryPlacements",
        "tags": "MemoryTags",
        "graph_mapper": "MemoryGraphMapper",
        "data_n_time_steps": "DataNTimeSteps",
        "n_key_map": "MemoryMachinePartitionNKeysMap"
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "time_period_map", "time_scale_factor", "machine_graph",
            "routing_info", "placements", "tags", "graph_mapper",
            "application_graph", "n_key_map", "data_n_time_steps"])
    def generate_data_specification(
            self, spec, placement, time_period_map, time_scale_factor,
            machine_graph, routing_info, placements, tags, graph_mapper,
            application_graph, n_key_map, data_n_time_steps):

        # reserve regions
        self._reserve_memory_regions(spec, machine_graph, n_key_map, self)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), time_period_map[self],
            time_scale_factor))

        # params
        self._write_param_region(spec, machine_graph, routing_info)

        # float params
        self._write_double_params_region(spec)

        # write the filter params
        self._write_filter_params(spec)

        # sdram edge
        self._write_sdram_edge_region(spec, machine_graph)

        # only write params if used
        self._write_profile_dsg(spec)

        # write up the bitfield builder data
        bit_field_utilities.write_bitfield_init_data(
            spec, self, machine_graph, routing_info,
            n_key_map, self.REGIONS.BIT_FIELD_BUILDER.value,
            self.REGIONS.POPULATION_TABLE.value,
            self.REGIONS.SYNAPTIC_MATRIX.value,
            self.REGIONS.DIRECT_MATRIX.value,
            self.REGIONS.BIT_FIELD_FILTER.value,
            self.REGIONS.BIT_FIELD_KEY_MAP.value)

        # Write the recording regions
        self._neuron_recorder.write_neuron_recording_region(
            spec, self.REGIONS.NEURON_RECORDING.value,
            Slice(self._drnl_index, self._drnl_index),
            data_n_time_steps)

        self._synapse_manager.write_data_spec(
            spec, graph_mapper.get_application_vertex(self),
            Slice(self._drnl_index, self._drnl_index), self, placement,
            machine_graph, application_graph, routing_info, graph_mapper,
            self.GLOBAL_WEIGHT_SCALE, time_period_map[self],
            self.REGIONS.SYNAPSE_PARAMS.value,
            self.REGIONS.POPULATION_TABLE.value,
            self.REGIONS.SYNAPTIC_MATRIX.value,
            self.REGIONS.DIRECT_MATRIX.value,
            self.REGIONS.SYNAPSE_DYNAMICS.value,
            self.REGIONS.CONNECTOR_BUILDER.value)

        self.__on_chip_generatable_offset = \
            self._synapse_manager.host_written_matrix_size

        self.__on_chip_generatable_size = \
            self._synapse_manager.on_chip_written_matrix_size

        # End the specification
        spec.end_specification()

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return [self.MOC_RECORDING_REGION_ID]

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.RECORDING.value, txrx)
