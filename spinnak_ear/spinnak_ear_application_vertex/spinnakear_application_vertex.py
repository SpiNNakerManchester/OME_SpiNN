from pacman.model.graphs.application import ApplicationEdge
from pacman.model.graphs.common import Slice, EdgeTrafficType
from pacman.model.graphs.impl.constant_sdram_machine_partition import \
    ConstantSDRAMMachinePartition
from pacman.model.graphs.machine.machine_sdram_edge import SDRAMMachineEdge
from pacman.model.partitioner_interfaces.\
    abstract_controls_destination_of_edges import \
    AbstractControlsDestinationOfEdges
from pacman.model.partitioner_interfaces.\
    abstract_controls_source_of_edges import \
    AbstractControlsSourceOfEdges
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.utilities import \
    globals_variables, helpful_functions
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.globals_variables import get_simulator
from spinnak_ear.spinnak_ear_edges.spinnaker_ear_machine_edge import \
    SpiNNakEarMachineEdge

from spynnaker.pyNN.models.abstract_models.\
    abstract_accepts_incoming_synapses import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.abstract_models.\
    abstract_sends_outgoing_synapses import \
    AbstractSendsOutgoingSynapses
from spynnaker.pyNN.models.common import AbstractSpikeRecordable, \
    AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

from pacman.model.graphs.application.\
    application_vertex import ApplicationVertex
from pacman.model.partitioner_interfaces.hand_over_to_vertex import \
    HandOverToVertex
from pacman.model.decorators.overrides import overrides
from pacman.executor.injection_decorator import inject_items

from spinnak_ear.spinnak_ear_machine_vertices.ome_machine_vertex import \
    OMEMachineVertex
from spinnak_ear.spinnak_ear_machine_vertices.drnl_machine_vertex import \
    DRNLMachineVertex
from spinnak_ear.spinnak_ear_machine_vertices.ihcan_machine_vertex import \
    IHCANMachineVertex
from spinnak_ear.spinnak_ear_machine_vertices.an_group_machine_vertex import \
    ANGroupMachineVertex

import numpy 
import math
import random
import logging

from spynnaker.pyNN.models.neuron.synaptic_manager import SynapticManager

logger = logging.getLogger(__name__)


class SpiNNakEarApplicationVertex(
        ApplicationVertex, AbstractAcceptsIncomingSynapses,
        SimplePopulationSettable, HandOverToVertex, AbstractChangableAfterRun,
        AbstractSpikeRecordable, AbstractNeuronRecordable,
        AbstractControlsDestinationOfEdges, AbstractControlsSourceOfEdges,
        AbstractSendsOutgoingSynapses):

    __slots__ = [
        # pynn model
        '_model',
        # bool flag for neuron param changes
        '_remapping_required',
        # ihcan vertices 
        "_ihcan_vertices",
        # drnl vertices
        "_drnl_vertices",
        # final agg verts (outgoing atoms)
        "_final_agg_vertices",
        # storing synapse dynamics
        "_synapse_dynamics",
        # fibres per.... something
        "_n_fibres_per_ihc",
        # recording spikes
        "_is_recording_spikes",
        # recording the attenuation value
        "_is_recording_moc",
        # recording the probability of the inner hair to spike
        "_is_recording_inner_hair_spike_prob",
        # the seed for the inner hair fibre
        "_ihcan_fibre_random_seed",
        # the number of columns / rows for aggregation tree
        "_n_group_tree_rows",
        # the synaptic manager to manage projections into drnl verts.
        "__synapse_manager",
        # the number of drnls there are.
        "_n_dnrls",
        # the number of agg verts which are final aggegation verts.
        "_n_final_agg_groups"
    ]

    # NOTES IHC = inner hair cell
    #       IHCan =  inner hair channel
    #       DRNL = middle ear filter
    #       OME ear fluid

    # error message for frequency
    FREQUENCY_ERROR = (
        "The input sampling frequency is too high for the chosen simulation " 
        "time scale. Please reduce Fs or increase the time scale factor in "
        "the config file")

    # error message for incorrect neurons map
    N_NEURON_ERROR = (
        "the number of neurons {} and the number of atoms  {} do not match")

    # app edge mc partition id
    MC_APP_EDGE_PARTITION_ID = "internal_mc"

    # app edge sdram partition id
    SDRAM_APP_EDGE_PARTITION_ID = "internal_sdram"

    # recording names
    SPIKES = "spikes"
    MOC = "moc"
    SPIKE_PROB = "inner_ear_spike_probability"

    # named flag
    _DRNL = "drnl"

    # whats recordable
    _RECORDABLES = [SPIKES, MOC, SPIKE_PROB]

    # recordable units NOTE RJ and ABS have no idea, but we're going with
    # meters for completeness on the MOC.
    _RECORDABLE_UNITS = {
        SPIKES: SPIKES,
        MOC: 'meters',
        SPIKE_PROB: "percentage"
    }

    # n recording regions
    _N_POPULATION_RECORDING_REGIONS = 1

    # random numbers
    _FINAL_ROW_N_ATOMS = 256
    MAX_TIME_SCALE_FACTOR_RATIO = 22050
    HSR_FLAG = 2
    MSR_FLAG = 1
    LSR_FLAG = 0

    N_SYNAPSE_TYPES = 2

    def __init__(
            self, n_neurons, constraints, label, model, profile):
        # Superclasses
        ApplicationVertex.__init__(self, label, constraints)
        AbstractAcceptsIncomingSynapses.__init__(self)
        SimplePopulationSettable.__init__(self)
        HandOverToVertex.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractSpikeRecordable.__init__(self)

        self._model = model
        self._profile = profile
        self._remapping_required = True
        self._synapse_dynamics = None
        self._ihcan_vertices = list()
        self._drnl_vertices = list()
        self._final_agg_vertices = list()
        self.__synapse_manager = SynapticManager(
            self.N_SYNAPSE_TYPES, None, None,
            globals_variables.get_simulator().config)

        # ear hair frequency bits in total per inner ear channel
        self._n_fibres_per_ihc = (
            self._model.n_lsr_per_ihc + self._model.n_msr_per_ihc +
            self._model.n_hsr_per_ihc)

        # number of columns needed for the aggregation tree
        atoms_per_row = self.calculate_atoms_per_row(
            self._model.n_channels, self._n_fibres_per_ihc,
            self._model.n_fibres_per_ihcan,
            self._model.max_input_to_aggregation_group)

        # ????????
        max_n_atoms_per_group_tree_row = (
            (self._model.max_input_to_aggregation_group **
             numpy.arange(1, atoms_per_row + 1)) *
            self._model.n_fibres_per_ihcan)

        # ????????????
        max_n_atoms_per_group_tree_row = \
            max_n_atoms_per_group_tree_row[
                max_n_atoms_per_group_tree_row <= self._FINAL_ROW_N_ATOMS]

        self._n_group_tree_rows = max_n_atoms_per_group_tree_row.size

        # recording flags
        self._is_recording_spikes = False
        self._is_recording_moc = False
        self._is_recording_inner_hair_spike_prob = False

        config = globals_variables.get_simulator().config
        self._time_scale_factor = helpful_functions.read_config_int(
            config, "Machine", "time_scale_factor")
        if (self._model.fs / self._time_scale_factor >
                self.MAX_TIME_SCALE_FACTOR_RATIO):
            raise Exception(self.FREQUENCY_ERROR)

        if self._model.param_file is not None:
            try:
                pre_gen_vars = numpy.load(self._model.param_file)
                self._n_atoms = pre_gen_vars['n_atoms']
                self._mv_index_list = pre_gen_vars['mv_index_list']
                self._parent_index_list = pre_gen_vars['parent_index_list']
                self._edge_index_list = pre_gen_vars['edge_index_list']
                self._ihc_seeds = pre_gen_vars['ihc_seeds']
                self._ome_indices = pre_gen_vars['ome_indices']
            except:
                self._n_atoms, self._n_dnrls, self._n_final_agg_groups = \
                    self.calculate_n_atoms(
                        atoms_per_row,
                        self._model.max_input_to_aggregation_group,
                        self._model.n_channels, self._model.n_ihc)
                # save fixed param file
                self._save_pre_gen_vars(self._model.param_file)
        else:
            self._n_atoms, self._n_dnrls, self._n_final_agg_groups = \
                self.calculate_n_atoms(
                    atoms_per_row, self._model.max_input_to_aggregation_group,
                    self._model.n_channels, self._model.n_ihc)

        #if self._n_atoms != n_neurons:
        #    raise ConfigurationException(
        #        self.N_NEURON_ERROR.format(n_neurons, self._n_atoms))

    @overrides(AbstractSendsOutgoingSynapses.get_out_going_size)
    def get_out_going_size(self):
        return self._n_final_agg_groups

    def get_out_going_slices(self):
        slices = list()
        starter = 0
        for _ in self._final_agg_vertices:
            slices.append(Slice(starter, starter))
            starter += 1
        return slices

    def get_in_coming_slices(self):
        slices = list()
        starter = 0
        for _ in self._drnl_vertices:
            slices.append(Slice(starter, starter))
            starter += 1
        return slices

    @overrides(AbstractControlsSourceOfEdges.get_pre_slice_for)
    def get_pre_slice_for(self, machine_vertex):
        if isinstance(machine_vertex, ANGroupMachineVertex):
            if machine_vertex.is_final_row:
                return Slice(machine_vertex.low_atom, machine_vertex.low_atom)
        raise Exception(
            "Why are you asking for a source outside of aggregation verts?!")

    @overrides(AbstractControlsDestinationOfEdges.get_post_slice_for)
    def get_post_slice_for(self, machine_vertex):
        if isinstance(machine_vertex, DRNLMachineVertex):
            return Slice(machine_vertex.drnl_index, machine_vertex.drnl_index)
        raise Exception(
            "why you asking for a destination atoms outside of the drnl "
            "verts!?")

    @overrides(AbstractAcceptsIncomingSynapses.get_in_coming_size)
    def get_in_coming_size(self):
        return self._n_dnrls

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        if target == "excitatory":
            return 0
        elif target == "inhibitory":
            return 1
        return None

    @overrides(
        AbstractControlsDestinationOfEdges.get_destinations_for_edge_from)
    def get_destinations_for_edge_from(
            self, app_edge, partition_id, graph_mapper,
            original_source_machine_vertex):
        if ((app_edge.pre_vertex != self and app_edge.post_vertex == self)
            and not isinstance(
                original_source_machine_vertex, OMEMachineVertex)):
            return self._drnl_vertices
        else:
            return []

    @overrides(AbstractControlsSourceOfEdges.get_sources_for_edge_from)
    def get_sources_for_edge_from(
            self, app_edge, partition_id, graph_mapper,
            original_source_machine_vertex):
        if ((app_edge.pre_vertex == self and app_edge.post_vertex != self)
                and isinstance(
                    original_source_machine_vertex, ANGroupMachineVertex)
                and original_source_machine_vertex.is_final_row):
            return [original_source_machine_vertex]
        else:
            return []

    @overrides(
        AbstractAcceptsIncomingSynapses.get_maximum_delay_supported_in_ms)
    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        return 1 * machine_time_step

    @overrides(AbstractAcceptsIncomingSynapses.add_pre_run_connection_holder)
    def add_pre_run_connection_holder(
            self, connection_holder, projection_edge, synapse_information):
        self.__synapse_manager.add_pre_run_connection_holder(
            connection_holder, projection_edge, synapse_information)

    def _save_pre_gen_vars(self, file_path):
        """
        saves params into a numpy file. 
        :param file_path: path to file to store stuff into
        :rtype: None 
        """
        numpy.savez_compressed(
            file_path, n_atoms=self._n_atoms,
            mv_index_list=self._mv_index_list,
            parent_index_list=self._parent_index_list,
            edge_index_list=self._edge_index_list,
            ihc_seeds=self._ihc_seeds,
            ome_indices=self._ome_indices)

    @overrides(AbstractAcceptsIncomingSynapses.set_synapse_dynamics)
    def set_synapse_dynamics(self, synapse_dynamics):
        self._synapse_dynamics = synapse_dynamics

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, machine_time_step, using_extra_monitor_cores,
            placements=None,  monitor_api=None, monitor_placement=None,
            monitor_cores=None, handle_time_out_configuration=True,
            fixed_routes=None):
        return self.__synapse_manager.get_connections_from_machine(
            transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, machine_time_step, using_extra_monitor_cores,
            DRNLMachineVertex.REGIONS.POPULATION_TABLE.value,
            DRNLMachineVertex.REGIONS.SYNAPTIC_MATRIX.value,
            DRNLMachineVertex.REGIONS.DIRECT_MATRIX.value,
            placements, monitor_api, monitor_placement, monitor_cores,
            handle_time_out_configuration, fixed_routes)

    @overrides(AbstractAcceptsIncomingSynapses.clear_connection_cache)
    def clear_connection_cache(self):
        self.__synapse_manager.clear_connection_cache()

    @overrides(SimplePopulationSettable.set_value)
    def set_value(self, key, value):
        SimplePopulationSettable.set_value(self, key, value)
        self._remapping_required = True

    def describe(self):
        """ Returns a human-readable description of the cell or synapse type.

        The output may be customised by specifying a different template\
        together with an associated template engine\
        (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template\
        context will be returned.
        """

        parameters = dict()
        for parameter_name in self._model.default_parameters:
            parameters[parameter_name] = self.get_value(parameter_name)

        context = {
            "name": self._model.model_name,
            "default_parameters": self._model.default_parameters,
            "default_initial_values": self._model.default_parameters,
            "parameters": parameters,
        }
        return context

    def _add_to_graph_components(
            self, machine_graph, graph_mapper, lo_atom, vertex,
            resource_tracker):
        machine_graph.add_vertex(vertex)
        graph_mapper.add_vertex_mapping(
            vertex, Slice(lo_atom, lo_atom), self)
        resource_tracker.allocate_constrained_resources(
            vertex.resources_required, vertex.constraints)
        return lo_atom + 1

    def _build_ome_vertex(
            self, machine_graph, graph_mapper, lo_atom, resource_tracker):
        """ builds the ome vertex
        
        :param machine_graph: machine graph
        :param graph_mapper: graph mapper
        :param lo_atom: lo atom to put into graph mapper slice
        :param resource_tracker: the resource tracker
        :return: the ome vertex and the new low atom
        """
        # build the ome machine vertex
        ome_vertex = OMEMachineVertex(
            self._model.audio_input, self._model.fs, self._model.n_channels,
            self._model.seq_size, profile=self._profile)

        # allocate resources and updater graphs
        new_lo_atom = self._add_to_graph_components(
            machine_graph, graph_mapper, lo_atom, ome_vertex, resource_tracker)
        return ome_vertex, new_lo_atom

    def _build_drnl_verts(
            self, machine_graph, graph_mapper, new_low_atom, resource_tracker,
            ome_vertex):
        """ build the drnl verts
        
        :param machine_graph: machine graph
        :param graph_mapper: graph mapper
        :param new_low_atom: the current low atom count for the graph mapper
        :param resource_tracker: the resource tracker for placement
        :param ome_vertex: the ome vertex to tie edges to
        :return: 
        """
        pole_index = 0
        for _ in range(self._model.n_channels):
            drnl_vertex = DRNLMachineVertex(
                self._model.pole_freqs[pole_index], self._model.fs,
                ome_vertex.n_data_points, pole_index, self._is_recording_moc,
                self._profile, self._model.seq_size, self.__synapse_manager,
                self, self._model.n_buffers_in_sdram_total)
            pole_index += 1
            new_low_atom = self._add_to_graph_components(
                machine_graph, graph_mapper, new_low_atom, drnl_vertex,
                resource_tracker)
            self._drnl_vertices.append(drnl_vertex)
        return new_low_atom

    def _build_edges_between_ome_drnls(
            self, ome_vertex, machine_graph, app_edge, graph_mapper):
        """ adds edges between the ome and the drnl vertices
        
        :param ome_vertex: the ome vertex
        :param machine_graph: the machine graph
        :param app_edge: the app edge covering all these edges
        :param graph_mapper: the graph mapper
        :return: 
        """
        for drnl_vert in self._drnl_vertices:
            edge = SpiNNakEarMachineEdge(ome_vertex, drnl_vert)
            machine_graph.add_edge(edge, ome_vertex.OME_PARTITION_ID)
            graph_mapper.add_edge_mapping(edge, app_edge)

    def _build_ihcan_vertices_and_sdram_edges(
            self, machine_graph, graph_mapper, new_low_atom,
            resource_tracker, app_edge, sdram_app_edge):
        """ builds the ihcan verts and adds edges from drnl to them
        
        :param machine_graph: machine graph
        :param graph_mapper: the graph mapper
        :param new_low_atom: the lo atom sued to keep the graph mapper happy
        :param resource_tracker: the resource tracker for placement
        :param app_edge: the app edge to link all mc machine edges to
        :param sdram_app_edge: the application sdram edge between drnl and 
        inchan to link all sdram machine edges to. 
        :return: iterable of ihcan verts
        """

        ichans = list()

        # generate ihc seeds
        n_ihcans = self._model.n_channels * self._model.n_ihc
        seed_index = 0
        random_range = numpy.arange(
            n_ihcans * IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX,
            dtype=numpy.uint32)
        numpy.random.seed(self._model.ihc_seeds_seed)
        ihc_seeds = numpy.random.choice(
            random_range,
            int(n_ihcans * IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX),
            replace=False)

        for drnl_vertex in self._drnl_vertices:
            machine_graph.add_outgoing_edge_partition(
                ConstantSDRAMMachinePartition(
                    drnl_vertex.DRNL_SDRAM_PARTITION_ID, drnl_vertex,
                    "sdram edge between drnl vertex {} and its "
                    "IHCANS".format(drnl_vertex.drnl_index)))

            fibres = []
            for _ in range(self._model.n_hsr_per_ihc):
                fibres.append(self.HSR_FLAG)
            for __ in range(self._model.n_msr_per_ihc):
                fibres.append(self.MSR_FLAG)
            for ___ in range(self._model.n_lsr_per_ihc):
                fibres.append(self.LSR_FLAG)

            random.seed(self._model.ihcan_fibre_random_seed)
            random.shuffle(fibres)

            for _ in range(self._model.n_ihc):

                # randomly pick fibre types
                chosen_indices = [
                    fibres.pop() for _ in range(self._model.n_fibres_per_ihcan)]

                vertex = IHCANMachineVertex(
                    self._model.resample_factor,
                    ihc_seeds[
                        seed_index:
                        seed_index +
                        IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX],
                    self._is_recording_spikes,
                    self._is_recording_inner_hair_spike_prob,
                    self._model.n_fibres_per_ihcan,
                    self._model.ear_index, self._profile, self._model.fs,
                    chosen_indices.count(self.LSR_FLAG),
                    chosen_indices.count(self.MSR_FLAG),
                    chosen_indices.count(self.HSR_FLAG),
                    self._model.max_n_fibres_per_ihcan,
                    drnl_vertex.n_data_points,
                    self._model.n_buffers_in_sdram_total,
                    self._model.seq_size)
                seed_index += IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX
                ichans.append(vertex)

                new_low_atom = self._add_to_graph_components(
                    machine_graph, graph_mapper, new_low_atom, vertex,
                    resource_tracker)

                # multicast
                mc_edge = SpiNNakEarMachineEdge(
                    drnl_vertex, vertex, EdgeTrafficType.MULTICAST)
                machine_graph.add_edge(mc_edge, drnl_vertex.DRNL_PARTITION_ID)
                graph_mapper.add_edge_mapping(mc_edge, app_edge)

                # sdram edge
                sdram_edge = SDRAMMachineEdge(
                    drnl_vertex, vertex,
                    "sdram between {} and {}".format(drnl_vertex, vertex),
                    DRNLMachineVertex.sdram_edge_size)
                machine_graph.add_edge(
                    sdram_edge, drnl_vertex.DRNL_SDRAM_PARTITION_ID)
                graph_mapper.add_edge_mapping(sdram_edge, sdram_app_edge)
        return ichans, new_low_atom

    def _build_aggregation_group_vertices_and_edges(
            self, ichan_vertices, machine_graph, graph_mapper,
            new_low_atom, resource_tracker, app_edge):

        to_process = ichan_vertices
        n_child_per_group = self._model.max_input_to_aggregation_group

        for row in range(self._n_group_tree_rows):
            aggregation_verts = list()
            n_row_angs = int(
                numpy.ceil(float(len(to_process)) / n_child_per_group))
            for an in range(n_row_angs):
                final_row_lo_atom = 0
                child_verts = to_process[
                    an * n_child_per_group:
                    an * n_child_per_group + n_child_per_group]

                # deduce n atoms of the ag node
                n_atoms = 0
                for child in child_verts:
                    n_atoms += child.n_atoms

                # build vert
                final_row = row == self._n_group_tree_rows - 1
                ag_vertex = ANGroupMachineVertex(
                    n_atoms, len(child_verts), final_row, final_row_lo_atom)
                if final_row:
                    self._final_agg_vertices.append(ag_vertex)
                    final_row_lo_atom += 1
                aggregation_verts.append(ag_vertex)

                # update stuff
                new_low_atom = self._add_to_graph_components(
                    machine_graph, graph_mapper, new_low_atom, ag_vertex,
                    resource_tracker)

                # add edges
                for child_vert in child_verts:
                    # sort out partition id
                    partition_id = IHCANMachineVertex.IHCAN_PARTITION_ID
                    if isinstance(child_vert, ANGroupMachineVertex):
                        partition_id = \
                            ANGroupMachineVertex.AN_GROUP_PARTITION_IDENTIFER

                    # add edge and mapping
                    mc_edge = SpiNNakEarMachineEdge(child_vert, ag_vertex)
                    machine_graph.add_edge(mc_edge, partition_id)
                    graph_mapper.add_edge_mapping(mc_edge, app_edge)

            to_process = aggregation_verts

    @inject_items({"machine_time_step": "MachineTimeStep",
                   "application_graph": "MemoryApplicationGraph"})
    @overrides(
        HandOverToVertex.create_and_add_to_graphs_and_resources,
        additional_arguments={"machine_time_step", "application_graph"}
    )
    def create_and_add_to_graphs_and_resources(
            self, resource_tracker, machine_graph, graph_mapper,
            machine_time_step, application_graph):

        mc_app_edge = ApplicationEdge(self, self)
        sdram_app_edge = ApplicationEdge(self, self, EdgeTrafficType.SDRAM)
        application_graph.add_edge(mc_app_edge, self.MC_APP_EDGE_PARTITION_ID)
        application_graph.add_edge(
            sdram_app_edge, self.SDRAM_APP_EDGE_PARTITION_ID)

        # atom tracker
        current_atom_count = 0

        # ome vertex
        ome_vertex, current_atom_count = self._build_ome_vertex(
            machine_graph, graph_mapper, current_atom_count, resource_tracker)

        # handle the drnl verts
        current_atom_count = self._build_drnl_verts(
            machine_graph, graph_mapper, current_atom_count, resource_tracker,
            ome_vertex)

        # handle edges between ome and drnls
        self._build_edges_between_ome_drnls(
            ome_vertex, machine_graph, mc_app_edge, graph_mapper)

        # build the ihcan verts.
        ichan_vertices, current_atom_count = (
            self._build_ihcan_vertices_and_sdram_edges(
                machine_graph, graph_mapper, current_atom_count,
                resource_tracker, mc_app_edge, sdram_app_edge))

        # build aggregation group verts and edges
        self._build_aggregation_group_vertices_and_edges(
            ichan_vertices, machine_graph, graph_mapper, current_atom_count,
            resource_tracker, mc_app_edge)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_atoms

    @staticmethod
    def calculate_n_atoms(
            n_group_tree_rows, max_input_to_aggregation_group, n_channels,
            n_ihc):
        # ome atom
        n_atoms = 1

        # dnrl atoms
        n_atoms += n_channels

        # ihcan atoms
        n_angs = n_channels * n_ihc
        n_atoms += n_angs

        # an group atoms
        for row_index in range(n_group_tree_rows):
            n_row_angs = int(
                numpy.ceil(float(n_angs) / max_input_to_aggregation_group))
            n_atoms += n_row_angs
            n_angs = n_row_angs
        return n_atoms, n_channels, n_angs

    @staticmethod
    def calculate_atoms_per_row(
            n_channels, n_fibres_per_ihc, n_fibres_per_ihcan,
            max_input_to_aggregation_group):
        return int(numpy.ceil(math.log(
            (n_channels * n_fibres_per_ihc) / n_fibres_per_ihcan,
            max_input_to_aggregation_group)))

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        self._remapping_required = False

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        return self._remapping_required

    @overrides(AbstractSpikeRecordable.is_recording_spikes)
    def is_recording_spikes(self):
        return self._is_recording_spikes

    @overrides(AbstractSpikeRecordable.get_spikes_sampling_interval)
    def get_spikes_sampling_interval(self):
        # TODO this needs fixing properly
        return get_simulator().machine_time_step

    @overrides(AbstractSpikeRecordable.clear_spike_recording)
    def clear_spike_recording(self, buffer_manager, placements, graph_mapper):
        for ihcan_vertex in self._ihcan_vertices:
            placement = placements.get_placement_of_vertex(ihcan_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p,
                IHCANMachineVertex.SPIKE_RECORDING_REGION_ID)

    @overrides(AbstractSpikeRecordable.get_spikes)
    def get_spikes(
            self, placements, graph_mapper, buffer_manager, machine_time_step):
        samples = list()
        for ihcan_vertex in self._ihcan_vertices:
            # Read the data recorded
            for fibre in ihcan_vertex.read_samples(
                    buffer_manager,
                    placements.get_placement_of_vertex(ihcan_vertex)):
                samples.append(fibre)
        return numpy.asarray(samples)

    @overrides(AbstractSpikeRecordable.set_recording_spikes)
    def set_recording_spikes(
            self, new_state=True, sampling_interval=None, indexes=None):
        self._is_recording_spikes = new_state
        self._remapping_required = not self.is_recording(self.SPIKES)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        return self._RECORDABLES

    @overrides(AbstractNeuronRecordable.clear_recording)
    def clear_recording(self, variable, buffer_manager, placements,
                        graph_mapper):
        if variable == self.MOC:
            for drnl_vertex in self._drnl_vertices:
                placement = placements.get_placement_of_vertex(drnl_vertex)
                buffer_manager.clear_recorded_data(
                    placement.x, placement.y, placement.p,
                    DRNLMachineVertex.MOC_RECORDING_REGION_ID)
        else:
            raise ConfigurationException(
                "Spinnakear does not support recording of {}".format(variable))

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        #TODO need to do this properly
        return 1

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(self, variable, new_state=True, sampling_interval=None,
                      indexes=None):
        if variable == self.MOC:
            self._remapping_required = not self.is_recording(variable)
            self._is_recording_moc = new_state
        else:
            raise ConfigurationException(
                "Spinnakear does not support recording of {}".format(variable))

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        if variable == self.SPIKES:
            return self._is_recording_spikes
        elif variable == self.MOC:
            return self._is_recording_moc
        else:
            raise ConfigurationException(
                "Spinnakear does not support recording of {}".format(variable))

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(self, variable, n_machine_time_steps, placements,
                 graph_mapper, buffer_manager, machine_time_step):
        if variable == self.MOC:
            samples = list()
            for drnl_vertex in self._drnl_vertices:
                placement = placements.get_placement_of_vertex(drnl_vertex)
                samples.append(drnl_vertex.read_moc_attenuation(
                    buffer_manager, placement))
            return numpy.asarray(samples)
        else:
            raise ConfigurationException(
                "Spinnakear does not support recording of {}".format(variable))
