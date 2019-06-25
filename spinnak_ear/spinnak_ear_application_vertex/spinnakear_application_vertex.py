from pacman.model.graphs.common import Slice, EdgeTrafficType
from pacman.model.graphs.impl.constant_sdram_machine_partition import \
    ConstantSDRAMMachinePartition
from pacman.model.graphs.machine.machine_sdram_edge import SDRAMMachineEdge
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.utilities import \
    globals_variables, helpful_functions
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.globals_variables import get_simulator

from spynnaker.pyNN.models.abstract_models.\
    abstract_accepts_incoming_synapses import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractSpikeRecordable, \
    AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

from pacman.model.graphs.application.\
    application_vertex import ApplicationVertex
from pacman.model.partitioner_interfaces.hand_over_to_vertex import \
    HandOverToVertex
from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.machine.machine_edge import MachineEdge
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
logger = logging.getLogger(__name__)


class SpiNNakEarApplicationVertex(
        ApplicationVertex, AbstractAcceptsIncomingSynapses,
        SimplePopulationSettable, HandOverToVertex, AbstractChangableAfterRun,
        AbstractSpikeRecordable, AbstractNeuronRecordable):

    __slots__ = [
        # pynn model
        '_model',
        # bool flag for neuron param changes
        '_remapping_required',
        # ihcan vertices 
        "_ihcan_vertices",
        # drnl vertices
        "_drnl_vertices",
        # storing synapse dynamics
        "_synapse_dynamics",
        # fibres per.... something
        "_n_fibres_per_ihc",
        #
        "_is_recording_spikes",
        #
        "_is_recording_moc",
        #
        "_ihcan_fibre_random_seed",
        # the number of columns / rows for aggregation tree
        "_n_group_tree_rows"
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

    # recording names
    SPIKES = "spikes"
    MOC = "moc"

    # named flag
    _DRNL = "drnl"

    # whats recordable
    _RECORDABLES = [SPIKES, MOC]

    # recordable units
    _RECORDABLE_UNITS = {
        SPIKES: SPIKES,
        MOC: '??????'
    }

    # n recording regions
    _N_POPULATION_RECORDING_REGIONS = 1

    # random numbers
    _N_SEEDS_PER_IHCAN_VERTEX = 4
    _FINAL_ROW_N_ATOMS = 256
    MAX_TIME_SCALE_FACTOR_RATIO = 22050
    HSR_FLAG = 2
    MSR_FLAG = 1
    LSR_FLAG = 0

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

        # ear hair frequency bits in total per inner ear channel
        self._n_fibres_per_ihc = (
            self._model.n_lsr_per_ihc + self._model.n_msr_per_ihc +
            self._model.n_hsr_per_ihc)

        # number of columns needed for the aggregation tree
        atoms_per_row = int(numpy.ceil(math.log(
            (self._model.n_channels * self._n_fibres_per_ihc) /
            self._model.n_fibres_per_ihcan,
            self._model.max_input_to_aggregation_group)))

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
                self._n_atoms = self._calculate_n_atoms(atoms_per_row)
                # save fixed param file
                self._save_pre_gen_vars(self._model.param_file)
        else:
            self._n_atoms = self._calculate_n_atoms(atoms_per_row)

        #if self._n_atoms != n_neurons:
        #    raise ConfigurationException(
        #        self.N_NEURON_ERROR.format(n_neurons, self._n_atoms))

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    @overrides(
        AbstractAcceptsIncomingSynapses.get_maximum_delay_supported_in_ms)
    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        return 1 * machine_time_step

    @overrides(AbstractAcceptsIncomingSynapses.add_pre_run_connection_holder)
    def add_pre_run_connection_holder(
            self, connection_holder, projection_edge, synapse_information):
        pass

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
        raise Exception("cant get connections from projections to here yet")

    @overrides(AbstractAcceptsIncomingSynapses.clear_connection_cache)
    def clear_connection_cache(self):
        pass

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
        resource_tracker.allocate_constrained_resources(
            vertex.resources_required, vertex.constraints)
        machine_graph.add_vertex(vertex)
        graph_mapper.add_vertex_mapping(
            vertex, Slice(lo_atom, lo_atom + 1), self)
        lo_atom += 1

    def _build_ome_vertex(
            self, machine_graph, graph_mapper, lo_atom, resource_tracker):
        # build the ome machine vertex
        ome_vertex = OMEMachineVertex(
            self._model.audio_input, self._model.fs, self._model.n_channels,
            time_scale=self._time_scale_factor, profile=False)

        # allocate resources and updater graphs
        self._add_to_graph_components(
            machine_graph, graph_mapper, lo_atom, ome_vertex, resource_tracker)
        return ome_vertex

    def _build_drnl_verts(
            self, machine_graph, graph_mapper, current_atom_count,
            resource_tracker, ome_vertex):
        """
        
        :param machine_graph: 
        :param graph_mapper: 
        :param current_atom_count: 
        :param resource_tracker: 
        :param ome_vertex: 
        :return: 
        """
        drnl_verts = list()
        pole_index = 0
        for _ in range(self._model.n_channels):
            drnl_vertex = DRNLMachineVertex(
                self._model.pole_freqs[pole_index], 0.0, self._model.fs,
                ome_vertex.n_data_points, pole_index, self._is_recording_moc,
                False)
            pole_index += 1
            self._add_to_graph_components(
                machine_graph, graph_mapper, current_atom_count, drnl_vertex,
                resource_tracker)
            drnl_verts.append(drnl_vertex)
        return drnl_verts

    def _build_edges_between_ome_drnls(
            self, ome_vertex, drnl_verts, machine_graph):
        """
        
        :param ome_vertex: 
        :param drnl_verts: 
        :param machine_graph: 
        :return: 
        """
        for drnl_vert in drnl_verts:
            edge = MachineEdge(ome_vertex, drnl_vert)
            machine_graph.add_edge(edge, ome_vertex.OME_PARTITION_ID)

    def _build_ihcan_vertices_and_sdram_edges(
            self, drnl_verts, machine_graph, graph_mapper, current_atom_count,
            resource_tracker):
        """ 
        
        :param drnl_verts: 
        :return: iterable of ihcan verts
        """

        ichans = list()

        # generate ihc seeds
        n_ihcans = self._model.n_channels * self._model.n_ihc
        seed_index = 0
        random_range = numpy.arange(
            n_ihcans * self._N_SEEDS_PER_IHCAN_VERTEX, dtype=numpy.uint32)
        numpy.random.seed(self._model.ihc_seeds_seed)
        ihc_seeds = numpy.random.choice(
            random_range, int(n_ihcans * self._N_SEEDS_PER_IHCAN_VERTEX),
            replace=False)

        for drnl_vertex in drnl_verts:
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
                        seed_index + self._N_SEEDS_PER_IHCAN_VERTEX],
                    self._is_recording_spikes,
                    self._model.n_fibres_per_ihcan,
                    self._model.ear_index, True, False, self._model.fs,
                    chosen_indices.count(self.LSR_FLAG),
                    chosen_indices.count(self.MSR_FLAG),
                    chosen_indices.count(self.HSR_FLAG),
                    self._model.max_n_fibres_per_ihcan,
                    drnl_vertex.n_data_points)
                seed_index += self._N_SEEDS_PER_IHCAN_VERTEX
                ichans.append(vertex)

                self._add_to_graph_components(
                    machine_graph, graph_mapper, current_atom_count, vertex,
                    resource_tracker)

                # multicast
                machine_graph.add_edge(
                    MachineEdge(
                        pre_vertex=drnl_vertex, post_vertex=vertex,
                        traffic_type=EdgeTrafficType.MULTICAST),
                    drnl_vertex.DRNL_PARTITION_ID)

                # sdram edge
                machine_graph.add_edge(
                    SDRAMMachineEdge(
                        pre_vertex=drnl_vertex, post_vertex=vertex,
                        label="sdram between {} and {}".format(
                            drnl_vertex, vertex),
                        sdram_size=DRNLMachineVertex.SDRAM_SIZE),
                    drnl_vertex.DRNL_SDRAM_PARTITION_ID)
        return ichans

    def _build_aggration_group_vertices_and_edges(
            self, ichan_vertices, machine_graph, graph_mapper,
            current_atom_count, resource_tracker):

        to_process = ichan_vertices
        n_child_per_group = self._model.max_input_to_aggregation_group

        for row in range(self._n_group_tree_rows):
            aggregation_verts = list()
            n_row_angs = int(
                numpy.ceil(float(len(to_process)) / n_child_per_group))
            for an in range(n_row_angs):
                child_verts = to_process[
                    an * n_child_per_group:
                    an * n_child_per_group + n_child_per_group]

                # deduce n atoms of the ag node
                n_atoms = 0
                for child in child_verts:
                    n_atoms += child.n_atoms

                # build vert
                ag_vertex = ANGroupMachineVertex(
                    n_atoms, row == self._n_group_tree_rows - 1)
                aggregation_verts.append(ag_vertex)

                # update stuff
                self._add_to_graph_components(
                    machine_graph, graph_mapper, current_atom_count, ag_vertex,
                    resource_tracker)

                # add edges
                for child_vert in child_verts:
                    partition_id = IHCANMachineVertex.IHCAN_PARTITION_ID
                    if isinstance(child_vert, ANGroupMachineVertex):
                        partition_id = \
                            ANGroupMachineVertex.AN_GROUP_PARTITION_IDENTIFER
                    machine_graph.add_edge(
                        MachineEdge(
                            child_vert, ag_vertex, EdgeTrafficType.MULTICAST),
                        partition_id)
            to_process = aggregation_verts

    @inject_items({"machine_time_step": "MachineTimeStep"})
    @overrides(
        HandOverToVertex.create_and_add_to_graphs_and_resources,
        additional_arguments={"machine_time_step"}
    )
    def create_and_add_to_graphs_and_resources(
            self, resource_tracker, machine_graph, graph_mapper,
            machine_time_step):
        # atom tracker
        current_atom_count = 0

        # ome vertex
        ome_vertex = self._build_ome_vertex(
            machine_graph, graph_mapper, current_atom_count, resource_tracker)

        # handle the drnl verts
        drnl_verts = self._build_drnl_verts(
            machine_graph, graph_mapper, current_atom_count, resource_tracker,
            ome_vertex)

        # handle edges between ome and drnls
        self._build_edges_between_ome_drnls(
            ome_vertex, drnl_verts, machine_graph)

        # build the ihcan verts.
        ichan_vertices = self._build_ihcan_vertices_and_sdram_edges(
            drnl_verts, machine_graph, graph_mapper, current_atom_count,
            resource_tracker)

        # build aggregation group verts and edges
        self._build_aggration_group_vertices_and_edges(
            ichan_vertices, machine_graph, graph_mapper, current_atom_count,
            resource_tracker)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_atoms

    def _calculate_n_atoms(self, n_group_tree_rows):
        # ome atom
        n_atoms = 1

        # dnrl atoms
        n_atoms += self._model.n_channels

        # ihcan atoms
        n_angs = self._model.n_channels * self._model.n_ihc
        n_atoms += n_angs

        # an group atoms
        for row_index in range(n_group_tree_rows):
            n_row_angs = int(
                numpy.ceil(
                    float(n_angs) /
                    self._model.max_input_to_aggregation_group))
            n_atoms += n_row_angs
            n_angs = n_row_angs
        return n_atoms

    @overrides(HandOverToVertex.source_vertices_from_edge)
    def source_vertices_from_edge(self, edge):
        """ returns vertices for connecting this projection

        :param edge: projection to connect to sources
        :return: the iterable of vertices to be sources of this projection
        """

    @overrides(HandOverToVertex.destination_vertices_from_edge)
    def destination_vertices_from_edge(self, edge):
        """ return vertices for connecting this projection

        :param edge: projection to connect to destinations
        :return: the iterable of vertices to be destinations of this \
        projection.
        """

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
