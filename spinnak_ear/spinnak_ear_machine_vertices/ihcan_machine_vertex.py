from pacman.model.graphs.common import EdgeTrafficType
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
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
from spinn_front_end_common.interface.buffer_management.buffer_models\
    .abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.utilities.utility_objs import ExecutableType, \
    ProvenanceDataItem
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
import math


class IHCANMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractReceiveBuffersToHost,
        AbstractEarProfiled, ProvidesProvenanceDataFromMachineImpl,
        AbstractMachineSupportsAutoPauseAndResume):
    """ A vertex that runs the IHCAN algorithm
    """

    __slots__ = [

        # which ear (left or right)
        "_ear_index",

        # ????
        "_re_sample_factor",

        # sample freq
        "_fs",

        # ?????????
        "_dt",

        # n atoms in this vertex (inner hair cells?)
        "_n_atoms",

        # n low freq hair cells in here
        "_n_lsr",

        # n medium freq hair cells in here
        "_n_msr",

        # n high freq hair cells in here
        "_n_hsr",

        # the seg size
        "_seq_size",

        # data points.....
        "_num_data_points",

        # seed for random
        "_seed",

        # the number of distinct buffers in the sdram
        "_n_buffers_in_sdram_total",

        # the ihcan recorder
        "_ihcan_neuron_recorder",

        # the slice of atoms for this ihcan vertex from the global atoms
        "_ihcan_atom_slice",

        # timer period
        "_timer_period"
    ]

    # converts voltage into release rate
    Z = 40e12

    # ?????????????
    _GMAXCA = 20e-9

    # cilia params with divide
    # ???????????????
    CILIA_RECIPS0 = 1.0 / 6e-9

    # ?????????????????????
    CILIA_RECIPS1 = 1.0 / 1e-9

    # other constants
    # ???????????????
    GU0 = (1e-10 + 6e-9 / (
        1 + math.exp(0.3e-9 / 6e-9) * (1 + math.exp(1e-9 / 1e-9))))
    # ???????????????
    EKP = (-0.08 + 0.04 * 0.1)

    # ???????????????
    IHCV = (2.1e-8 * EKP + GU0 * 0.1) / (GU0 + 2.1e-8)

    # ???????????????
    M_ICA_CURR = (1.0 / (1.0 + math.exp(-100.0 * IHCV) * (1.0 / 400.0)))

    # ???????????????
    _M_ICA_CUR_POW = pow(M_ICA_CURR, 2)

    # ???????????????
    CA_CURR_LSR = ((_GMAXCA * _M_ICA_CUR_POW) * (IHCV - 0.066)) * 200e-6

    # ???????????????
    CA_CURR_MSR = ((_GMAXCA * _M_ICA_CUR_POW) * (IHCV - 0.066)) * 350e-6

    # ???????????????
    CA_CURR_HSR = ((_GMAXCA * _M_ICA_CUR_POW) * (IHCV - 0.066)) * 500e-6

    # inner ear stuff
    # ???????????????
    _MSR = 4

    # ???????????????
    _Y = 15

    # ???????????????
    _L = 150

    # ???????????????
    _R = 300

    # ???????????????
    _X = 300

    # ???????????????
    _KT0LSR = Z * math.pow(CA_CURR_LSR, 3)

    # ???????????????
    _KT0MSR = Z * math.pow(CA_CURR_MSR, 3)

    # ???????????????
    _KT0HSR = Z * math.pow(CA_CURR_HSR, 3)

    # ???????????????
    AN_CLEFT_LSR = (_KT0LSR * _Y * _MSR) / (_Y * (_L + _R) + _KT0LSR * _L)

    # ???????????????
    AN_CLEFT_MSR = (_KT0MSR * _Y * _MSR) / (_Y * (_L + _R) + _KT0MSR * _L)

    # ???????????????
    AN_CLEFT_HSR = (_KT0HSR * _Y * _MSR) / (_Y * (_L + _R) + _KT0HSR * _L)

    # ???????????????
    AN_AVAIL_LSR = round((AN_CLEFT_LSR * (_L + _R)) / _KT0LSR)

    # ???????????????
    AN_AVAIL_MSR = round((AN_CLEFT_MSR * (_L + _R)) / _KT0MSR)

    # ???????????????
    AN_AVAIL_HSR = round((AN_CLEFT_HSR * (_L + _R)) / _KT0HSR)

    # ???????????????
    AN_REPRO_LSR = (AN_CLEFT_LSR * _R) / _X

    # ???????????????
    AN_REPRO_MSR = (AN_CLEFT_MSR * _R) / _X

    # ???????????????
    AN_REPRO_HSR = (AN_CLEFT_HSR * _R) / _X

    # The number of params for the parameters region
    # 1. resampling factor, 2. n fibres, 3. seg size
    # 4. number of sdram buffers. 5. num_lsr. 6. num_msr. 7. num_hsr.
    # 8. my_key
    _N_PARAMETERS = 8

    # the number of params for the sdram edge region
    _N_SDRAM_EDGE_PARAMS = 1

    # the number of params in the dt params region
    _N_DT_PARAMS = 2

    # the number of params in the synapse params region
    _N_SYANPSE_PARAMS = 5

    # the number of params in the cilia_constants_param region
    _N_CILIA_PARAMS = 2

    # the number of params in the inner ear params region
    _N_INNER_EAR_PARAM_PARAMS = 16

    # the number of elements in the seeds
    N_SEEDS_PER_IHCAN_VERTEX = 4

    # unknown what this magic number is
    MAGIC_1 = 1000.0

    # IHCAN partition id
    IHCAN_PARTITION_ID = "IHCANData"

    # dsg regions
    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('CILIA_PARAMS', 2),
               ('INNER_EAR_PARAMS', 3),
               ('DT_BASED_PARAMS', 4),
               ('RANDOM_SEEDS', 5),
               ('NEURON_RECORDING', 6),
               ('SDRAM_EDGE', 7),
               ('PROFILE', 8),
               ('PROVENANCE', 9)])

    # provenance items
    EXTRA_PROVENANCE_DATA_ENTRIES = Enum(
        value="EXTRA_PROVENANCE_DATA_ENTRIES",
        names=[("N_SIMULATION_TICKS", 0),
               ("SEG_INDEX", 1),
               ("DATA_READ_COUNT", 2),
               ("DATA_WRITE_COUNT_SPIKES", 3),
               ("DATA_WRITE_COUNT_SPIKE_PROB", 4),
               ("MC_RX_COUNT", 5),
               ("MC_TRANSMISSION_COUNT", 6),
               ("N_PROVENANCE_ELEMENTS", 7)])

    # recording regions
    RECORDING_REGIONS = Enum(
        value="RECORDING_REGIONS",
        names=[("SPIKE_RECORDING_REGION_ID", 0),
               ("SPIKE_PROBABILITY_REGION_ID", 1),
               ("N_RECORDING_REGIONS", 2)]
    )

    # recording names
    SPIKES = "spikes"
    SPIKE_PROB = "inner_ear_spike_probability"

    # the params recordable from a drnl vertex
    RECORDABLES = [SPIKES, SPIKE_PROB]

    # recordable units NOTE RJ and ABS have no idea, but we're going with
    # meters for completeness on the MOC.
    RECORDABLE_UNITS = {
        SPIKES: SPIKES,
        SPIKE_PROB: "%"
    }

    # fibres error
    N_FIBRES_ERROR = (
        "Only {} fibres can be modelled per IHCAN, currently requesting {} "
        "lsr, {}msr, {}hsr")

    # message when recording not complete
    RECORDING_WARNING = (
        "recording not complete, reduce Fs or disable RT!\n recorded output "
        "length:{}, expected length:{} at placement:{},{},{}")

    def __init__(
            self, resample_factor, seed, n_fibres, ear_index, profile, fs,
            n_lsr, n_msr, n_hsr, n_buffers_in_sdram_total, seq_size,
            ihcan_neuron_recorder, ihcan_atom_slice, timer_period):
        """ constructor

        :param resample_factor: resample factor
        :param seed: the seed used for its random number generator in SpiNNaker
        :param n_fibres: how many fibres to simulate
        :param ear_index: which ear its based on
        :param profile: bool flag for profiling
        :param fs: sample freq
        :param n_lsr: number of low freq hair cells
        :param n_msr: number of med freq hair cells
        :param n_hsr: number of high freq hair cells
        :param n_buffers_in_sdram_total: the total number of sdram buffers in \
        the sdram edge
        :param seg_size: the seq size
        :param ihcan_neuron_recorder: recorder for the ihcan recordings
        :param ihcan_atom_slice: the slice of atoms for the ihcan vertex from \
        the global.
        """

        MachineVertex.__init__(self, label="IHCAN Node", constraints=None)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.PROFILE.value)
        ProvidesProvenanceDataFromMachineImpl.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractGeneratesDataSpecification.__init__(self)
        AbstractReceiveBuffersToHost.__init__(self)

        self._ihcan_neuron_recorder = ihcan_neuron_recorder
        self._ihcan_atom_slice = ihcan_atom_slice
        self._ear_index = ear_index

        self._re_sample_factor = resample_factor
        self._fs = fs
        self._dt = 1.0 / self._fs
        self._n_atoms = n_fibres
        self._n_buffers_in_sdram_total = n_buffers_in_sdram_total
        self._seq_size = seq_size
        self._timer_period = timer_period

        if n_lsr + n_msr + n_hsr > n_fibres:
            raise Exception(
                self.N_FIBRES_ERROR.format(n_fibres, n_lsr, n_msr, n_hsr))

        self._n_lsr = n_lsr
        self._n_msr = n_msr
        self._n_hsr = n_hsr
        self._seed = seed

    @staticmethod
    def get_matrix_scalar_data_types():
        return {IHCANMachineVertex.SPIKE_PROB: DataType.FLOAT_32}

    @staticmethod
    def get_matrix_output_data_types():
        return {IHCANMachineVertex.SPIKE_PROB: DataType.FLOAT_32}

    @overrides(AbstractMachineSupportsAutoPauseAndResume.my_local_time_period)
    def my_local_time_period(self, simulator_time_step):
        return self._timer_period

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.REGIONS.PROVENANCE.value

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        simulation_ticks = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_SIMULATION_TICKS.value]
        seq_index = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.SEG_INDEX.value]
        data_read_count = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.DATA_READ_COUNT.value]
        data_write_count_spikes = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.DATA_WRITE_COUNT_SPIKES.value]
        data_write_count_spike_prob = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.
            DATA_WRITE_COUNT_SPIKE_PROB.value]
        mc_rx_count = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.MC_RX_COUNT.value]
        mc_transmission_count = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.MC_TRANSMISSION_COUNT.value]

        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "n simulation ticks executed"),
            simulation_ticks))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "last seq index"), seq_index))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "how many reads occurred"), data_read_count))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "how many writes occurred for spikes"),
            data_write_count_spikes))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "how many writes occurred for spike probs"),
            data_write_count_spike_prob))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "how many multicast packets received"),
            mc_rx_count))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "how many multicast packets sent"),
            mc_transmission_count))
        return provenance_items

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):

        # system region
        sdram = constants.SYSTEM_BYTES_REQUIREMENT

        # params + cilia + inner ear + seeds + sdram edge + DT elements +
        # synapse
        sdram_params = (
            self._N_PARAMETERS + self._N_CILIA_PARAMS + self._N_DT_PARAMS +
            + self._N_INNER_EAR_PARAM_PARAMS + self._N_SDRAM_EDGE_PARAMS +
            self.N_SEEDS_PER_IHCAN_VERTEX)
        sdram += sdram_params * constants.WORD_TO_BYTE_MULTIPLIER

        # profile region
        sdram += self._profile_size()

        # provenance region
        sdram += self.get_provenance_data_size(
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value)

        # recording region
        # recording stuff
        sdram += self._ihcan_neuron_recorder.get_sdram_usage_in_bytes(
            self._ihcan_atom_slice)
        variable_sdram = self._ihcan_neuron_recorder.get_variable_sdram_usage(
            self._ihcan_atom_slice)

        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=variable_sdram,
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

    def _fill_in_sdram_edge_region(self, spec, machine_graph):
        sdram_partition = None
        for edge in machine_graph.get_edges_ending_at_vertex(self):
            if isinstance(edge.pre_vertex, DRNLMachineVertex):
                if edge.traffic_type == EdgeTrafficType.SDRAM:
                    sdram_partition = \
                        machine_graph.get_outgoing_partition_for_edge(edge)
        spec.switch_write_focus(self.REGIONS.SDRAM_EDGE.value)
        spec.write_value(sdram_partition.sdram_base_address)

    def _fill_in_parameter_region(self, spec, routing_info):

        # write parameters
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the spike resample factor
        spec.write_value(self._re_sample_factor)

        # write n fibres
        spec.write_value(self._n_atoms)

        # write seg size
        spec.write_value(self._seq_size)

        # write n sdram buffers
        spec.write_value(self._n_buffers_in_sdram_total)

        # Write number of spontaneous fibres
        spec.write_value(int(self._n_lsr))
        spec.write_value(int(self._n_msr))
        spec.write_value(int(self._n_hsr))

        # Write the routing key
        key = routing_info.get_first_key_from_pre_vertex(
            self, self.IHCAN_PARTITION_ID)
        spec.write_value(key)

    def _fill_in_cilia_parameter_region(self, spec):
        """ writes cilia recips constants

        :param spec: dsg
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.CILIA_PARAMS.value)
        spec.write_value(self.CILIA_RECIPS0, DataType.FLOAT_32)
        spec.write_value(self.CILIA_RECIPS1, DataType.FLOAT_32)

    def _fill_in_inner_ear_parameter_region(self, spec):
        """ writes the inner ear constants

        :param spec: dsg
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.INNER_EAR_PARAMS.value)
        spec.write_value(self.AN_CLEFT_LSR, DataType.FLOAT_32)
        spec.write_value(self.AN_CLEFT_MSR, DataType.FLOAT_32)
        spec.write_value(self.AN_CLEFT_HSR, DataType.FLOAT_32)
        spec.write_value(self.AN_AVAIL_LSR, DataType.FLOAT_32)
        spec.write_value(self.AN_AVAIL_MSR, DataType.FLOAT_32)
        spec.write_value(self.AN_AVAIL_HSR, DataType.FLOAT_32)
        spec.write_value(self.AN_REPRO_LSR, DataType.FLOAT_32)
        spec.write_value(self.AN_REPRO_MSR, DataType.FLOAT_32)
        spec.write_value(self.AN_REPRO_HSR, DataType.FLOAT_32)
        spec.write_value(self.IHCV, DataType.FLOAT_32)
        spec.write_value(self.M_ICA_CURR, DataType.FLOAT_32)
        spec.write_value(self.EKP, DataType.FLOAT_32)
        spec.write_value(self.CA_CURR_LSR, DataType.FLOAT_32)
        spec.write_value(self.CA_CURR_MSR, DataType.FLOAT_32)
        spec.write_value(self.CA_CURR_HSR, DataType.FLOAT_32)
        spec.write_value((1.0 / (0xffffffff + 1.0)), DataType.FLOAT_32)

    def _fill_in_dt_param_region(self, spec):
        """ writes the dt based constants

        :param spec: the specification writer
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.DT_BASED_PARAMS.value)
        spec.write_value(self._dt, DataType.FLOAT_32)
        spec.write_value(self.Z, DataType.FLOAT_32)

    def _fill_in_seed_region(self, spec):
        """ stores seeds needed for the RNG on spinnaker

        :param spec: dsg.
        :rtype: None
        """

        # Write the seed
        spec.switch_write_focus(self.REGIONS.RANDOM_SEEDS.value)
        data = numpy.array(self._seed, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

    def _reserve_memory_regions(self, spec):
        """ reserve memory regions

        :param spec: the data spec
        :return: None
        """

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve the parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value,
            self._N_PARAMETERS * constants.WORD_TO_BYTE_MULTIPLIER, "params")

        # Reserve the cilia params region
        spec.reserve_memory_region(
            self.REGIONS.CILIA_PARAMS.value,
            self._N_CILIA_PARAMS * constants.WORD_TO_BYTE_MULTIPLIER,
            "cilia params")

        # Reserve the inner ear params region
        spec.reserve_memory_region(
            self.REGIONS.INNER_EAR_PARAMS.value,
            self._N_INNER_EAR_PARAM_PARAMS * constants.WORD_TO_BYTE_MULTIPLIER,
            "inner ear params")

        # Reserve the dt based params
        spec.reserve_memory_region(
            self.REGIONS.DT_BASED_PARAMS.value,
            self._N_DT_PARAMS * constants.WORD_TO_BYTE_MULTIPLIER,
            "dt based params")

        spec.reserve_memory_region(
            self.REGIONS.RANDOM_SEEDS.value,
            self.N_SEEDS_PER_IHCAN_VERTEX * constants.WORD_TO_BYTE_MULTIPLIER,
            "random seed region")

        spec.reserve_memory_region(
            self.REGIONS.SDRAM_EDGE.value,
            self._N_SDRAM_EDGE_PARAMS * constants.WORD_TO_BYTE_MULTIPLIER,
            "sdram edge region")

        # reserve provenance data region
        self.reserve_provenance_data_region(spec)

        # reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.NEURON_RECORDING.value,
            self._ihcan_neuron_recorder.get_static_sdram_usage(
                self._ihcan_atom_slice))

        # profiler region
        self._reserve_profile_memory_regions(spec)

    @inject_items({
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "machine_graph": "MemoryMachineGraph",
        "time_period_map": "MachineTimeStepMap",
        "time_scale_factor": "TimeScaleFactor",
        "data_n_time_steps": "DataNTimeSteps",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "routing_info", "tags", "placements", "machine_graph",
            "time_period_map", "time_scale_factor", "data_n_time_steps"])
    def generate_data_specification(
            self, spec, placement, routing_info, tags, placements,
            machine_graph, time_period_map, time_scale_factor,
            data_n_time_steps):

        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), time_period_map[self],
            time_scale_factor))

        # fill in the parameters region
        self._fill_in_parameter_region(spec, routing_info)

        # fill in the cilia params
        self._fill_in_cilia_parameter_region(spec)

        # fill in the inner ear params
        self._fill_in_inner_ear_parameter_region(spec)

        # fill in the dt based params region
        self._fill_in_dt_param_region(spec)

        # fill in the random seed region
        self._fill_in_seed_region(spec)

        # fill in the sdram edge data region
        self._fill_in_sdram_edge_region(spec, machine_graph)

        # Write the recording regions
        self._ihcan_neuron_recorder.write_neuron_recording_region(
            spec, self.REGIONS.NEURON_RECORDING.value,
            self._ihcan_atom_slice, data_n_time_steps)

        # Write profile regions
        self._write_profile_dsg(spec)

        # End the specification
        spec.end_specification()

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return self._ihcan_neuron_recorder.recorded_region_ids

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.NEURON_RECORDING.value, txrx)
