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

from spinn_front_end_common.abstract_models import AbstractChangableAfterRun, \
    AbstractCanReset
from spinn_front_end_common.abstract_models.\
    abstract_application_supports_auto_pause_and_resume import \
    AbstractApplicationSupportsAutoPauseAndResume
from spinn_front_end_common.utilities import globals_variables
from spinn_front_end_common.utilities.constants import \
    MICRO_TO_SECOND_CONVERSION
from spinn_front_end_common.utilities.exceptions import ConfigurationException

from spynnaker.pyNN.models.abstract_models import AbstractContainsUnits, \
    AbstractPopulationSettable
from spynnaker.pyNN.models.abstract_models.\
    abstract_accepts_incoming_synapses import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.abstract_models.\
    abstract_sends_outgoing_synapses import \
    AbstractSendsOutgoingSynapses
from spynnaker.pyNN.models.common import AbstractSpikeRecordable, \
    AbstractNeuronRecordable, NeuronRecorder
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable
from spynnaker.pyNN.models.neuron.synapse_dynamics import SynapseDynamicsStatic
from spynnaker.pyNN.models.neuron.synaptic_manager import SynapticManager

from pacman.model.graphs.application.\
    application_vertex import ApplicationVertex
from pacman.model.partitioner_interfaces.hand_over_to_vertex import \
    HandOverToVertex
from pacman.model.decorators.overrides import overrides
from pacman.executor.injection_decorator import inject_items

from spinnak_ear.spinnak_ear_edges.spinnaker_ear_machine_edge import \
    SpiNNakEarMachineEdge
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

logger = logging.getLogger(__name__)


class SpiNNakEarApplicationVertex(
        ApplicationVertex, AbstractAcceptsIncomingSynapses,
        SimplePopulationSettable, HandOverToVertex, AbstractChangableAfterRun,
        AbstractSpikeRecordable, AbstractNeuronRecordable,
        AbstractControlsDestinationOfEdges, AbstractControlsSourceOfEdges,
        AbstractSendsOutgoingSynapses, AbstractCanReset, AbstractContainsUnits,
        AbstractApplicationSupportsAutoPauseAndResume):

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
        # the seed for the inner hair fibre
        "_ihcan_fibre_random_seed",
        # the number of columns / rows for aggregation tree
        "_n_group_tree_rows",
        # the synaptic manager to manage projections into drnl verts.
        "__synapse_manager",
        # the number of drnls there are.
        "_n_dnrls",
        # the number of agg verts which are final aggregation verts.
        "_n_final_agg_groups",
        # the pole frequencies
        "_pole_freqs",
        # The timer period for the fast components
        "_timer_period"
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

    # error message for get units
    GET_UNIT_ERROR = "do not know what to do with variable {} for get units"

    # error message if getting source outside aggregation verts
    PRE_SLICE_ERROR = (
        "Why are you asking for a source outside of aggregation verts?!")

    # error message if getting destination verts outside drnls.
    POST_SLICE_ERROR = (
        "why you asking for a destination atoms outside of the drnl verts!?")

    # error for processing plastic synapses
    PLASTIC_SYNAPSE_ERROR = (
        "The SpiNNaear cannot handle plastic synapses at the moment, "
        "complain to the SpiNNaker software team if this is a problem.")

    # error message for being asked to clear recording of a param we dont know
    CLEAR_RECORDING_ERROR = "Spinnakear does not support recording of {}"

    # error message for set recording of a variable we dont know about
    RECORDING_ERROR = "Spinnakear does not support recording of {}"

    # error message for sampling interval
    SAMPLING_INTERVAL_ERROR = "do not know how to handle variable {}"

    # error message for incorrect neurons map
    N_NEURON_ERROR = (
        "the number of neurons {} and the number of atoms  {} do not match")

    # app edge mc partition id
    MC_APP_EDGE_PARTITION_ID = "internal_mc"

    # app edge sdram partition id
    SDRAM_APP_EDGE_PARTITION_ID = "internal_sdram"

    # green wood function from https://en.wikipedia.org/wiki/Greenwood_function
    # constant below and mapped to variable names

    # green wood constants for human cochlea hearing frequency mapping
    # A is a scaling constant between the characteristic frequency and the
    # upper frequency limit of the species
    GREEN_WOOD_HUMAN_CONSTANT_A = 165.4

    # a is the slope of the straight-line portion of the frequency-position
    # curve, which has shown to be conserved throughout all investigated
    # species after scaling the length of the cochlea
    GREEN_WOOD_HUMAN_CONSTANT_ALPHA = 2.1

    # K is a constant of integration that represents the divergence from the
    # log nature of the curve and is determined by the lower frequency
    # audible limit in the species.
    GREEN_WOOD_HUMAN_CONSTANT_K = 0.88

    # n recording regions
    _N_POPULATION_RECORDING_REGIONS = 1

    # random numbers
    _FINAL_ROW_N_ATOMS = 256
    MAX_TIME_SCALE_FACTOR_RATIO = 22050

    # flags for sorting out random fibres. might be a enum
    HSR_FLAG = 2
    MSR_FLAG = 1
    LSR_FLAG = 0

    # how many synapse types this binary supports
    N_SYNAPSE_TYPES = 2

    # these curve values are built from profiling the IHCAN cores to deduce
    # performance.
    CURVE_ONE = 18.12
    CURVE_TWO = 10.99

    # max audio frequency supported
    DEFAULT_MAX_AUDIO_FREQUENCY = 20000

    # biggest number of neurons for the ear model
    FULL_EAR_HAIR_FIBERS = 30000.0

    # min audio frequency supported
    DEFAULT_MIN_AUDIO_FREQUENCY = 30

    def __init__(
            self, n_neurons, constraints, label, model, profile,
            time_scale_factor):
        # Superclasses
        ApplicationVertex.__init__(self, label, constraints)
        AbstractAcceptsIncomingSynapses.__init__(self)
        SimplePopulationSettable.__init__(self)
        HandOverToVertex.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractSpikeRecordable.__init__(self)
        AbstractNeuronRecordable.__init__(self)

        self._model = model
        self._profile = profile
        self._remapping_required = True
        self._synapse_dynamics = None
        self._n_fibres_per_ihc = None
        self._n_group_tree_rows = None
        self._ihcan_vertices = list()
        self._drnl_vertices = list()
        self._final_agg_vertices = list()
        self.__synapse_manager = SynapticManager(
            self.N_SYNAPSE_TYPES, None, None,
            globals_variables.get_simulator().config)

        # calculate n fibres per ihcan core
        sample_time = time_scale_factor / self._model.fs

        # how many channels
        self._n_channels = int(
            self.get_out_going_size() / self._model.n_fibres_per_ihc)

        # process pole freqs
        self._pole_freqs = self._process_pole_freqs()

        # how many fibres / atoms ran on each ihcan core
        self._n_fibres_per_ihcan_core = self.fibres_per_ihcan_core(sample_time)

        # process all the other internal numbers
        atoms_per_row = self.process_internal_numbers()

        # read in param file if needed
        self._process_param_file(atoms_per_row)

        # recording stuff
        self._drnl_neuron_recorder = NeuronRecorder(
            DRNLMachineVertex.RECORDABLES,
            DRNLMachineVertex.get_matrix_scalar_data_types(),
            DRNLMachineVertex.get_matrix_output_data_types(), self._n_dnrls)

        self._ihcan_neuron_recorder = NeuronRecorder(
            IHCANMachineVertex.RECORDABLES,
            IHCANMachineVertex.get_matrix_scalar_data_types(),
            IHCANMachineVertex.get_matrix_output_data_types(),
            self._n_dnrls * self._n_fibres_per_ihcan_core *
            self._model.seq_size)

        # bool for if state has changed.
        self._change_requires_mapping = True
        self._change_requires_neuron_parameters_reload = False
        self._change_requires_data_generation = False
        self._has_reset_last = True

        # safety check
        if self._n_atoms != n_neurons:
            raise ConfigurationException(
                self.N_NEURON_ERROR.format(n_neurons, self._n_atoms))

        # safety stuff
        if (self._model.fs / time_scale_factor >
                self.MAX_TIME_SCALE_FACTOR_RATIO):
            raise Exception(self.FREQUENCY_ERROR)

        # write timer period
        self._timer_period = (
            MICRO_TO_SECOND_CONVERSION *
            self._model.seq_size / self._model.fs)

    @staticmethod
    def fibres_per_ihcan_core(sample_time):
        # how many fibras / atoms ran on each ihcan core
        return abs(int(
            math.floor(
                ((sample_time / MICRO_TO_SECOND_CONVERSION) -
                 SpiNNakEarApplicationVertex.CURVE_ONE) /
                SpiNNakEarApplicationVertex.CURVE_TWO)))

    @overrides(AbstractAcceptsIncomingSynapses.gen_on_machine)
    def gen_on_machine(self, vertex_slice):
        return self.__synapse_manager.gen_on_machine(vertex_slice)

    @overrides(
        AbstractApplicationSupportsAutoPauseAndResume.
        my_variable_local_time_period)
    def my_variable_local_time_period(
            self, default_machine_time_step, variable):
        if variable == DRNLMachineVertex.MOC:
            return default_machine_time_step
        else:
            return self._timer_period

    def reset_to_first_timestep(self):
        # Mark that reset has been done, and reload state variables
        self._has_reset_last = True
        self._change_requires_neuron_parameters_reload = False

        # If synapses change during the run,
        if self._synapse_manager.synapse_dynamics.changes_during_run:
            self._change_requires_data_generation = True

    def get_units(self, variable):
        if variable in DRNLMachineVertex.RECORDABLES:
            return DRNLMachineVertex.RECORDABLE_UNITS[variable]
        elif variable in IHCANMachineVertex.RECORDABLES:
            return IHCANMachineVertex.RECORDABLE_UNITS[variable]
        else:
            raise Exception(self.GET_UNIT_ERROR.format(variable))

    def _process_param_file(self, atoms_per_row):
        if self._model.param_file is not None:
            try:
                pre_gen_vars = numpy.load(self._model.param_file)
                self._n_atoms = pre_gen_vars['n_atoms']
                self._mv_index_list = pre_gen_vars['mv_index_list']
                self._parent_index_list = pre_gen_vars['parent_index_list']
                self._edge_index_list = pre_gen_vars['edge_index_list']
                self._ihc_seeds = pre_gen_vars['ihc_seeds']
                self._ome_indices = pre_gen_vars['ome_indices']
            except Exception:
                self._n_atoms, self._n_dnrls, self._n_final_agg_groups = \
                    self.calculate_n_atoms_for_each_vertex_type(
                        atoms_per_row,
                        self._model.max_input_to_aggregation_group,
                        self._n_channels, self._model.n_fibres_per_ihc,
                        self._model.seq_size)
                # save fixed param file
                self._save_pre_gen_vars(self._model.param_file)
        else:
            self._n_atoms, self._n_dnrls, self._n_final_agg_groups = \
                self.calculate_n_atoms_for_each_vertex_type(
                    atoms_per_row, self._model.max_input_to_aggregation_group,
                    self._n_channels, self._model.n_fibres_per_ihc,
                    self._model.seq_size)

    def process_internal_numbers(self):

        # ear hair frequency bits in total per inner ear channel
        self._n_fibres_per_ihc = (
            self._model.n_lsr_per_ihc + self._model.n_msr_per_ihc +
            self._model.n_hsr_per_ihc)

        # number of columns needed for the aggregation tree
        atoms_per_row = self.calculate_atoms_per_row(
            self._n_channels, self._n_fibres_per_ihc,
            self._n_fibres_per_ihcan_core,
            self._model.max_input_to_aggregation_group)

        # ????????
        max_n_atoms_per_group_tree_row = (
            (self._model.max_input_to_aggregation_group **
             numpy.arange(1, atoms_per_row + 1)) *
            self._n_fibres_per_ihcan_core)

        # ????????????
        max_n_atoms_per_group_tree_row = \
            max_n_atoms_per_group_tree_row[
                max_n_atoms_per_group_tree_row <= self._FINAL_ROW_N_ATOMS]

        self._n_group_tree_rows = max_n_atoms_per_group_tree_row.size
        return atoms_per_row

    def _process_pole_freqs(self):
        if self._model.pole_freqs is None:
            if self._model.fs > 2 * self.DEFAULT_MAX_AUDIO_FREQUENCY:  # use
                # the greenwood mapping
                pole_freqs = (
                    numpy.flipud([self.GREEN_WOOD_HUMAN_CONSTANT_A * (10 ** (
                        self.GREEN_WOOD_HUMAN_CONSTANT_ALPHA * numpy.linspace(
                            [0], [1], self._n_channels)) -
                            self.GREEN_WOOD_HUMAN_CONSTANT_K)]))

            # don't want alias frequencies so we use a capped log scale map
            else:
                max_power = min([numpy.log10(self.fs / 2.), numpy.log10(
                    self.DEFAULT_MAX_AUDIO_FREQUENCY)])
                pole_freqs = numpy.flipud(
                    numpy.logspace(
                        numpy.log10(self.DEFAULT_MIN_AUDIO_FREQUENCY),
                        max_power, self._n_channels))
        else:
            pole_freqs = self._model.pole_freqs
        return pole_freqs[0]

    @overrides(AbstractSendsOutgoingSynapses.get_out_going_size)
    def get_out_going_size(self):
        return (int(
            self.FULL_EAR_HAIR_FIBERS * float(self._model.scale) /
            self._model.n_fibres_per_ihc) * self._model.n_fibres_per_ihc)

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
        raise Exception(self.PRE_SLICE_ERROR)

    @overrides(AbstractControlsDestinationOfEdges.get_post_slice_for)
    def get_post_slice_for(self, machine_vertex):
        if isinstance(machine_vertex, DRNLMachineVertex):
            return Slice(machine_vertex.drnl_index, machine_vertex.drnl_index)
        raise Exception(self.POST_SLICE_ERROR)

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
    def get_maximum_delay_supported_in_ms(self, default_machine_time_step):
        return self.__synapse_manager.get_maximum_delay_supported_in_ms(
            default_machine_time_step)

    @overrides(AbstractAcceptsIncomingSynapses.add_pre_run_connection_holder)
    def add_pre_run_connection_holder(
            self, connection_holder, projection_edge, synapse_information):
        self.__synapse_manager.add_pre_run_connection_holder(
            connection_holder, projection_edge, synapse_information)

    def _save_pre_gen_vars(self, file_path):
        """ saves params into a numpy file.
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
        if not isinstance(synapse_dynamics, SynapseDynamicsStatic):
            raise Exception(self.PLASTIC_SYNAPSE_ERROR)
        self._synapse_dynamics = synapse_dynamics

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, local_time_step_map,
            using_extra_monitor_cores,
            placements=None,  monitor_api=None, monitor_placement=None,
            monitor_cores=None, handle_time_out_configuration=True,
            fixed_routes=None):
        return self.__synapse_manager.get_connections_from_machine(
            transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, local_time_step_map,
            using_extra_monitor_cores,
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

    @overrides(AbstractPopulationSettable.get_value)
    def get_value(self, key):
        if hasattr(self._model, key):
            return getattr(self._model, key)
        raise Exception("Population {} does not have parameter {}".format(
            self, key))

    def _add_to_graph_components(
            self, machine_graph, graph_mapper, slice, vertex,
            resource_tracker):
        """ adds the vertex to all the graph components and resources

        :param machine_graph: machine graph
        :param graph_mapper: graph mapper
        :param slice: slice
        :param vertex: machien vertex
        :param resource_tracker: resource tracker
        :rtype: None
        """

        machine_graph.add_vertex(vertex)
        graph_mapper.add_vertex_mapping(vertex, slice, self)
        resource_tracker.allocate_constrained_resources(
            vertex.resources_required, vertex.constraints)

    def _build_ome_vertex(
            self, machine_graph, graph_mapper, lo_atom, resource_tracker,
            timer_period):
        """ builds the ome vertex

        :param machine_graph: machine graph
        :param graph_mapper: graph mapper
        :param lo_atom: lo atom to put into graph mapper slice
        :param resource_tracker: the resource tracker
        :param timer_period: the timer period for all machine verts based on\
        the ear vertex
        :return: the ome vertex and the new low atom
        """
        # build the ome machine vertex
        ome_vertex = OMEMachineVertex(
            self._model.audio_input, self._model.fs, self._n_channels,
            self._model.seq_size, timer_period, self._profile)

        # allocate resources and updater graphs
        self._add_to_graph_components(
            machine_graph, graph_mapper, Slice(lo_atom, lo_atom), ome_vertex,
            resource_tracker)
        return ome_vertex, lo_atom + 1

    def _build_drnl_verts(
            self, machine_graph, graph_mapper, new_low_atom, resource_tracker,
            ome_vertex, timer_period):
        """ build the drnl verts

        :param machine_graph: machine graph
        :param graph_mapper: graph mapper
        :param new_low_atom: the current low atom count for the graph mapper
        :param resource_tracker: the resource tracker for placement
        :param ome_vertex: the ome vertex to tie edges to
        :param timer_period: the timer period for all machine verts based on\
        the ear vertex
        :return: new low atom count
        """
        pole_index = 0
        for _ in range(self._n_channels):
            drnl_vertex = DRNLMachineVertex(
                self._pole_freqs[pole_index], self._model.fs,
                ome_vertex.n_data_points, pole_index, self._profile,
                self._model.seq_size, self.__synapse_manager, self,
                self._model.n_buffers_in_sdram_total,
                self._drnl_neuron_recorder, timer_period)
            pole_index += 1
            self._add_to_graph_components(
                machine_graph, graph_mapper, Slice(new_low_atom, new_low_atom),
                drnl_vertex,  resource_tracker)
            new_low_atom += 1
            self._drnl_vertices.append(drnl_vertex)
        return new_low_atom

    def _build_edges_between_ome_drnls(
            self, ome_vertex, machine_graph, app_edge, graph_mapper):
        """ adds edges between the ome and the drnl vertices

        :param ome_vertex: the ome vertex
        :param machine_graph: the machine graph
        :param app_edge: the app edge covering all these edges
        :param graph_mapper: the graph mapper
        :rtype: None
        """
        for drnl_vert in self._drnl_vertices:
            edge = SpiNNakEarMachineEdge(ome_vertex, drnl_vert)
            machine_graph.add_edge(edge, ome_vertex.OME_PARTITION_ID)
            graph_mapper.add_edge_mapping(edge, app_edge)

    def _build_ihcan_vertices_and_sdram_edges(
            self, machine_graph, graph_mapper, new_low_atom,
            resource_tracker, app_edge, sdram_app_edge, timer_period):
        """ builds the ihcan verts and adds edges from drnl to them

        :param machine_graph: machine graph
        :param graph_mapper: the graph mapper
        :param new_low_atom: the lo atom sued to keep the graph mapper happy
        :param resource_tracker: the resource tracker for placement
        :param app_edge: the app edge to link all mc machine edges to
        :param sdram_app_edge: the application sdram edge between drnl and \
        inchan to link all sdram machine edges to.
        :param timer_period: the timer period for all machine verts based on\
        the ear vertex
        :return: iterable of ihcan verts
        """

        ihcans = list()

        # generate ihc seeds
        n_ihcans = self._n_channels * self._model.n_fibres_per_ihc
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

            for _ in range(
                    int(self._model.n_fibres_per_ihc /
                        self._n_fibres_per_ihcan_core)):

                # randomly pick fibre types
                chosen_indices = [
                    fibres.pop() for _ in range(self._n_fibres_per_ihcan_core)]

                ihcan_slice = Slice(
                    new_low_atom, new_low_atom + (
                        self._n_fibres_per_ihcan_core *
                        self._model.seq_size) - 1)

                vertex = IHCANMachineVertex(
                    self._model.resample_factor,
                    ihc_seeds[
                        seed_index:
                        seed_index +
                        IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX],
                    self._n_fibres_per_ihcan_core,
                    self._model.ear_index, self._profile, self._model.fs,
                    chosen_indices.count(self.LSR_FLAG),
                    chosen_indices.count(self.MSR_FLAG),
                    chosen_indices.count(self.HSR_FLAG),
                    self._model.n_buffers_in_sdram_total,
                    self._model.seq_size, self._ihcan_neuron_recorder,
                    ihcan_slice, timer_period)

                # update indexes
                new_low_atom += ihcan_slice.n_atoms
                seed_index += IHCANMachineVertex.N_SEEDS_PER_IHCAN_VERTEX

                # add to list of ihcans
                ihcans.append(vertex)

                self._add_to_graph_components(
                    machine_graph, graph_mapper, ihcan_slice, vertex,
                    resource_tracker)

                # multicast
                mc_edge = SpiNNakEarMachineEdge(
                    drnl_vertex, vertex, EdgeTrafficType.MULTICAST)
                machine_graph.add_edge(mc_edge, drnl_vertex.DRNL_PARTITION_ID)
                graph_mapper.add_edge_mapping(mc_edge, app_edge)

                # sdram edge
                sdram_edge = SDRAMMachineEdge(
                    drnl_vertex, vertex, drnl_vertex.sdram_edge_size,
                    "sdram between {} and {}".format(drnl_vertex, vertex))
                machine_graph.add_edge(
                    sdram_edge, drnl_vertex.DRNL_SDRAM_PARTITION_ID)
                graph_mapper.add_edge_mapping(sdram_edge, sdram_app_edge)
        return ihcans, new_low_atom

    def _build_aggregation_group_vertices_and_edges(
            self, machine_graph, graph_mapper,
            new_low_atom, resource_tracker, app_edge):

        to_process = list()
        to_process.extend(self._ihcan_vertices)
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

                # build silce for an node
                an_slice = Slice(new_low_atom, new_low_atom + n_atoms - 1)
                new_low_atom += n_atoms

                # build vert
                final_row = row == self._n_group_tree_rows - 1
                ag_vertex = ANGroupMachineVertex(
                    n_atoms, len(child_verts), final_row, final_row_lo_atom,
                    row)
                if final_row:
                    self._final_agg_vertices.append(ag_vertex)
                    final_row_lo_atom += 1
                aggregation_verts.append(ag_vertex)

                # update stuff
                self._add_to_graph_components(
                    machine_graph, graph_mapper, an_slice, ag_vertex,
                    resource_tracker)

                # add edges
                for child_vert in child_verts:
                    # sort out partition id
                    partition_id = IHCANMachineVertex.IHCAN_PARTITION_ID
                    if isinstance(child_vert, ANGroupMachineVertex):
                        partition_id = \
                            ANGroupMachineVertex.AN_GROUP_PARTITION_IDENTIFIER

                    # add edge and mapping
                    mc_edge = SpiNNakEarMachineEdge(child_vert, ag_vertex)
                    machine_graph.add_edge(mc_edge, partition_id)
                    graph_mapper.add_edge_mapping(mc_edge, app_edge)

            to_process = aggregation_verts

    @inject_items({"application_graph": "MemoryApplicationGraph"})
    @overrides(
        HandOverToVertex.create_and_add_to_graphs_and_resources,
        additional_arguments={"application_graph"}
    )
    def create_and_add_to_graphs_and_resources(
            self, resource_tracker, machine_graph, graph_mapper,
            application_graph):

        mc_app_edge = ApplicationEdge(self, self)
        sdram_app_edge = ApplicationEdge(self, self, EdgeTrafficType.SDRAM)
        application_graph.add_edge(mc_app_edge, self.MC_APP_EDGE_PARTITION_ID)
        application_graph.add_edge(
            sdram_app_edge, self.SDRAM_APP_EDGE_PARTITION_ID)

        # atom tracker
        current_atom_count = 0

        timer_period = (
            MICRO_TO_SECOND_CONVERSION * self._model.seq_size / self._model.fs)

        # ome vertex
        ome_vertex, current_atom_count = self._build_ome_vertex(
            machine_graph, graph_mapper, current_atom_count, resource_tracker,
            timer_period)

        # handle the drnl verts
        current_atom_count = self._build_drnl_verts(
            machine_graph, graph_mapper, current_atom_count, resource_tracker,
            ome_vertex, timer_period)

        # handle edges between ome and drnls
        self._build_edges_between_ome_drnls(
            ome_vertex, machine_graph, mc_app_edge, graph_mapper)

        # build the ihcan verts.
        self._ihcan_vertices, current_atom_count = (
            self._build_ihcan_vertices_and_sdram_edges(
                machine_graph, graph_mapper, current_atom_count,
                resource_tracker, mc_app_edge, sdram_app_edge, timer_period))

        # build aggregation group verts and edges
        self._build_aggregation_group_vertices_and_edges(
            machine_graph, graph_mapper, current_atom_count, resource_tracker,
            mc_app_edge)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_atoms

    @staticmethod
    def calculate_n_atoms_for_each_vertex_type(
            n_group_tree_rows, max_input_to_aggregation_group, n_channels,
            n_ihc, seq_size):
        # ome atom
        n_atoms = 1

        # dnrl atoms
        n_atoms += n_channels

        # ihcan atoms
        n_angs = n_channels * n_ihc
        n_atoms += (n_angs * seq_size)

        # an group atoms
        for row_index in range(n_group_tree_rows):
            n_row_angs = int(
                numpy.ceil(float(n_angs) / max_input_to_aggregation_group))
            n_atoms += (n_row_angs * (
                (row_index + 1) ** max_input_to_aggregation_group))
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
        return self._ihcan_neuron_recorder.is_recording(
            IHCANMachineVertex.SPIKES)

    @overrides(AbstractSpikeRecordable.get_spikes_sampling_interval)
    def get_spikes_sampling_interval(
            self, graph_mapper, local_time_period_map):
        return self._ihcan_neuron_recorder.get_neuron_sampling_interval(
            IHCANMachineVertex.SPIKES, self._ihcan_vertices[0],
            local_time_period_map)

    @overrides(AbstractSpikeRecordable.clear_spike_recording)
    def clear_spike_recording(self, buffer_manager, placements, graph_mapper):
        for ihcan_vertex in self._ihcan_vertices:
            placement = placements.get_placement_of_vertex(ihcan_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p,
                IHCANMachineVertex.SPIKE_RECORDING_REGION_ID)

    @overrides(AbstractSpikeRecordable.get_spikes)
    def get_spikes(
            self, placements, graph_mapper, buffer_manager,
            local_timer_period_map):
        return self._ihcan_neuron_recorder.get_spikes(
            self._label, buffer_manager,
            IHCANMachineVertex.RECORDING_REGIONS.SPIKE_RECORDING_REGION_ID
                .value,
            placements, graph_mapper, self, local_timer_period_map)

    @overrides(AbstractSpikeRecordable.set_recording_spikes)
    def set_recording_spikes(
            self, default_machine_time_step, new_state=True,
            sampling_interval=None, indexes=None):
        self.set_recording(
            IHCANMachineVertex.SPIKES, self._timer_period, new_state,
            sampling_interval, indexes)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        recordables = list()
        # don't take the drnl spikes, as there's only 1 api for spikes, and the
        # drnl spikes are only there for the recording limitations
        recordables.append(DRNLMachineVertex.MOC)
        recordables.extend(IHCANMachineVertex.RECORDABLES)
        return recordables

    @overrides(AbstractSpikeRecordable.get_spike_machine_vertices)
    def get_spike_machine_vertices(self, graph_mapper):
        return self._ihcan_vertices

    @overrides(AbstractNeuronRecordable.get_machine_vertices_for)
    def get_machine_vertices_for(self, variable, graph_mapper):
        if variable == DRNLMachineVertex.MOC:
            return self._drnl_vertices
        else:
            return self._ihcan_vertices

    @overrides(AbstractNeuronRecordable.clear_recording)
    def clear_recording(self, variable, buffer_manager, placements,
                        graph_mapper):
        if variable == DRNLMachineVertex.MOC:
            for drnl_vertex in self._drnl_vertices:
                placement = placements.get_placement_of_vertex(drnl_vertex)
                buffer_manager.clear_recorded_data(
                    placement.x, placement.y, placement.p,
                    DRNLMachineVertex.MOC_RECORDING_REGION_ID.value)
        if variable == self.SPIKES:
            for ihcan_vertex in self._ihcan_vertices:
                placement = placements.get_placement_of_vertex(ihcan_vertex)
                buffer_manager.clear_recorded_data(
                    placement.x, placement.y, placement.p,
                    (IHCANMachineVertex.RECORDING_REGIONS
                     .SPIKE_RECORDING_REGION_ID.value))
        if variable == self.SPIKE_PROB:
            for ihcan_vertex in self._ihcan_vertices:
                placement = placements.get_placement_of_vertex(ihcan_vertex)
                buffer_manager.clear_recorded_data(
                    placement.x, placement.y, placement.p,
                    (IHCANMachineVertex.RECORDING_REGIONS
                     .SPIKE_PROBABILITY_REGION_ID.value))
        else:
            raise ConfigurationException(
                self.CLEAR_RECORDING_ERROR.format(variable))

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(
            self, variable, graph_mapper, local_time_period_map):
        if variable == DRNLMachineVertex.MOC:
            return self._drnl_neuron_recorder.get_neuron_sampling_interval(
                variable, self._drnl_vertices[0], local_time_period_map)
        elif variable in IHCANMachineVertex.RECORDABLES:
            return self._ihcan_neuron_recorder.get_neuron_sampling_interval(
                variable, self._ihcan_vertices[0], local_time_period_map)
        else:
            raise Exception(self.SAMPLING_INTERVAL_ERROR.format(variable))

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(
            self, variable, default_machine_time_step, new_state=True,
            sampling_interval=None, indexes=None):
        self._change_requires_mapping = not self.is_recording(variable)
        if variable == DRNLMachineVertex.MOC:
            self._drnl_neuron_recorder.set_recording(
                variable, sampling_interval, indexes, self,
                default_machine_time_step, new_state)
        elif variable in IHCANMachineVertex.RECORDABLES:
            self._ihcan_neuron_recorder.set_recording(
                variable, sampling_interval, indexes, self,
                default_machine_time_step, new_state)
        else:
            raise ConfigurationException(self.RECORDING_ERROR.format(variable))

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        if variable == DRNLMachineVertex.MOC:
            self._drnl_neuron_recorder.is_recording(variable)
        elif variable in IHCANMachineVertex.RECORDABLES:
            self._ihcan_neuron_recorder.is_recording(variable)
        else:
            raise ConfigurationException(self.RECORDING_ERROR.format(variable))

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(
            self, variable, n_machine_time_steps, placements, graph_mapper,
            buffer_manager, local_time_period_map):
        if variable == DRNLMachineVertex.MOC:
            return self._drnl_neuron_recorder.get_matrix_data(
                self._label, buffer_manager,
                DRNLMachineVertex.MOC_RECORDABLE_REGION_ID,
                placements, graph_mapper, self, variable,
                n_machine_time_steps, local_time_period_map)
        elif variable == IHCANMachineVertex.SPIKE_PROB:
            return self._ihcan_neuron_recorder.get_matrix_data(
                self._label, buffer_manager,
                IHCANMachineVertex.RECORDING_REGIONS.
                SPIKE_PROBABILITY_REGION_ID.value,
                placements, graph_mapper, self, variable,
                n_machine_time_steps, local_time_period_map)
        elif variable == IHCANMachineVertex.SPIKES:
            return self._ihcan_neuron_recorder.get_spikes(
                self._label, buffer_manager,
                IHCANMachineVertex.RECORDING_REGIONS.
                SPIKE_RECORDING_REGION_ID.value,
                placements, graph_mapper, self, n_machine_time_steps)
        else:
            raise ConfigurationException(self.RECORDING_ERROR.format(variable))

    def get_sampling_interval(self, sample_size_window):
        return (
            (self._timer_period * sample_size_window) *
            MICRO_TO_SECOND_CONVERSION)
