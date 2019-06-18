from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.utilities import \
    globals_variables, helpful_functions
from spinn_front_end_common.utilities import \
    constants as front_end_common_constants
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
from pacman.model.resources import \
    ResourceContainer, ConstantSDRAM, CPUCyclesPerTickResource, DTCMResource
from pacman.model.graphs.machine.machine_edge import MachineEdge
from pacman.model.constraints.placer_constraints import SameChipAsConstraint
from pacman.executor.injection_decorator import inject_items

from spinn_utilities.progress_bar import ProgressBar


from data_specification.enums.data_type import DataType

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

data_partition_dict = {
    'drnl': 'DRNLData',
    'ome': 'OMEData'}


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
        "_n_fibres_per_ihc"
    ]

    # NOTES IHC = inner hair cell
    #       IHCan =  inner hair channel
    #       DRNL = middle ear filter
    #       OME ear fluid

    FREQUENCY_ERROR = (
        "The input sampling frequency is too high for the chosen simulation " 
        "time scale. Please reduce Fs or increase the time scale factor in "
        "the config file")

    SPIKES = "spikes"
    MOC = "moc"

    # named flag
    _DRNL = "drnl"

    # The data type of each data element
    _DATA_ELEMENT_TYPE = DataType.FLOAT_64

    # The data type of the data count
    _DATA_COUNT_TYPE = DataType.UINT32

    # The data type of the keys
    _KEY_ELEMENT_TYPE = DataType.UINT32

    # recording id
    SPIKE_RECORDING_REGION_ID = 0

    _RECORDABLES = [SPIKES, MOC]
    _RECORDABLE_UNITS = {
        SPIKES: SPIKES,
        MOC: ''
    }

    # n recording regions
    _N_POPULATION_RECORDING_REGIONS = 1

    _MAX_N_ATOMS_PER_CORE = 2
    _FINAL_ROW_N_ATOMS = 256
    MAX_TIME_SCALE_FACTOR_RATIO = 22050

    def __init__(
            self, n_neurons, constraints, label, model):
        # Superclasses
        ApplicationVertex.__init__(self, label, constraints)
        AbstractAcceptsIncomingSynapses.__init__(self)
        SimplePopulationSettable.__init__(self)
        HandOverToVertex.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractSpikeRecordable.__init__(self)

        self._model = model
        self._remapping_required = True
        self._synapse_dynamics = None
        self._ihcan_vertices = list()
        self._drnl_vertices = list()

        # ear hair frequency bits in total per inner ear channel
        self._n_fibres_per_ihc = (
            self._model.n_lsr_per_ihc + self._model.n_msr_per_ihc +
            self._model.n_hsr_per_ihc)

        # data per moc recording
        self._data_size_bytes = (
            (self._model.audio_input.size * self._DATA_ELEMENT_TYPE.size) +
            self._DATA_COUNT_TYPE.size)

        # ?????
        n_group_tree_rows = int(numpy.ceil(math.log(
            (self._model.n_channels * self._n_fibres_per_ihc) /
            self._model.n_fibres_per_ihcan, self._MAX_N_ATOMS_PER_CORE)))

        # ????????
        self._max_n_atoms_per_group_tree_row = (
            (self._MAX_N_ATOMS_PER_CORE **
             numpy.arange(1, n_group_tree_rows + 1)) *
            self._model.n_fibres_per_ihcan)

        # ????????????
        self._max_n_atoms_per_group_tree_row = \
            self._max_n_atoms_per_group_tree_row[
                self._max_n_atoms_per_group_tree_row <=
                self._FINAL_ROW_N_ATOMS]

        self._n_group_tree_rows = self._max_n_atoms_per_group_tree_row.size

        # recording flags
        self._is_recording_spikes = False
        self._is_recording_moc = False

        self._seed_index = 0
        self._pole_index = 0

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
                (self._n_atoms, self._mv_index_list, self._parent_index_list,
                 self._edge_index_list, self._ihc_seeds, self._ome_indices) = \
                    self._calculate_n_atoms(self._n_group_tree_rows)
                # save fixed param file
                self._save_pre_gen_vars(self._model.param_file)
        else:
            (self._n_atoms, self._mv_index_list, self._parent_index_list,
             self._edge_index_list, self._ihc_seeds, self._ome_indices) = \
                self._calculate_n_atoms(self._n_group_tree_rows)

        if self._n_atoms != n_neurons:
            raise ConfigurationException(
                "the number of neurons and the number of atoms do not match")

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
            placements=None, data_receiver=None,
            sender_extra_monitor_core_placement=None,
            extra_monitor_cores_for_router_timeout=None,
            handle_time_out_configuration=True, fixed_routes=None):
        raise Exception("cant get connections from projections to here yet")

    @overrides(AbstractAcceptsIncomingSynapses.clear_connection_cache)
    def clear_connection_cache(self):
        pass

    @overrides(ApplicationVertex.get_resources_used_by_atoms)
    def get_resources_used_by_atoms(self, vertex_slice):
        vertex_label = self._mv_index_list[vertex_slice.lo_atom]
        if vertex_label == "ome":
            sdram_resource_bytes = (9*4) + (6*8) + self._data_size_bytes
            drnl_vertices = [i for i in self._mv_index_list if i == self.DRNL]
            sdram_resource_bytes += (
                len(drnl_vertices) * self._KEY_ELEMENT_TYPE.size)

        elif vertex_label == "mack":
            sdram_resource_bytes = 2*4 + 4 * self._KEY_ELEMENT_TYPE.size

        elif vertex_label == self._DRNL:
            sdram_resource_bytes = 14*4
            sdram_resource_bytes += 512 * 12  # key mask tab
            sdram_resource_bytes += 8 * 8
            sdram_resource_bytes += 256  # max n bytes for conn_lut
            if self._is_recording_moc:
                sdram_resource_bytes += self._data_size_bytes

        # elif vertex_label == "ihc":
        elif "ihc" in vertex_label:
            if self._is_recording_spikes:
                sdram_resource_bytes = (
                    15 * 4 + 1 * self._KEY_ELEMENT_TYPE.size +
                    self._model.n_fibres_per_ihcan * numpy.ceil(
                        self._model.audio_input.size / 8.) * 4)
            else:
                sdram_resource_bytes = 15*4 + 1 * self._KEY_ELEMENT_TYPE.size
        else:  # angroup
            child_vertices = (
                [self._mv_list[vertex_index] for vertex_index
                 in self._parent_index_list[vertex_slice.lo_atom]])
            n_child_keys = len(child_vertices)
            sdram_resource_bytes = 5*4 + 12 * n_child_keys

        container = ResourceContainer(
            sdram=ConstantSDRAM(
                sdram_resource_bytes +
                front_end_common_constants.SYSTEM_BYTES_REQUIREMENT + 8),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container

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

    @inject_items({"machine_time_step": "MachineTimeStep"})
    @overrides(
        ApplicationVertex.create_machine_vertex,
        additional_arguments={"machine_time_step"}
    )
    def create_and_add_to_graphs_and_resources(
            self, resource_tracker, machine_graph, graph_mapper,
            machine_time_step):

        # lookup relevant mv type, parent mv and edges associated with this atom
        mv_type = self._mv_index_list[vertex_slice.lo_atom]
        parent_mvs = self._parent_index_list[vertex_slice.lo_atom]

        # ensure lowest parent index (ome) will be first in list
        parent_mvs.sort()
        mv_edges = self._edge_index_list[vertex_slice.lo_atom]

        if mv_type == 'ome':
            vertex = OMEMachineVertex(
                self._model.audio_input, self._model.fs,
                self._model.n_channels, time_scale=self._time_scale_factor,
                profile=False)
            vertex.add_constraint(EarConstraint())

        elif mv_type == 'drnl':
            for parent_index,parent in enumerate(parent_mvs):
                # first parent will be ome
                if parent_index in self._ome_indices:
                    ome = self._mv_list[parent]
                    vertex = DRNLMachineVertex(
                        ome, self._model.pole_freqs[self._pole_index], 0.0,
                        is_recording=self._is_recording_moc,
                        profile=False, drnl_index=self._pole_index)
                    self._pole_index += 1
                else:  # will be a mack vertex
                    self._mv_list[parent].register_mack_processor(vertex)
                    vertex.register_parent_processor(self._mv_list[parent])
            vertex.add_constraint(EarConstraint())

        elif 'ihc' in mv_type:#mv_type == 'ihc':
            for parent in parent_mvs:
                n_lsr = int(mv_type[-3])
                n_msr = int(mv_type[-2])
                n_hsr = int(mv_type[-1])
                vertex = IHCANMachineVertex(
                    self._mv_list[parent], 1,
                    self._ihc_seeds[self._seed_index:self._seed_index + 4],
                    self._is_recording_spikes,
                    ear_index=self._model.ear_index, bitfield=True,
                    profile=False, n_lsr=n_lsr, n_msr=n_msr, n_hsr=n_hsr)
                self._seed_index += 4

                # ensure placement is on the same chip as the parent DRNL
                vertex.add_constraint(
                    SameChipAsConstraint(self._mv_list[parent]))

        else:  # an_group
            child_vertices = [
                self._mv_list[vertex_index] for vertex_index in parent_mvs]
            row_index = int(mv_type[-1])
            max_n_atoms = self._max_n_atoms_per_group_tree_row[row_index]

            if len(mv_edges) == 0:
                # final row AN group
                is_final_row = True
            else:
                is_final_row = False

            vertex = ANGroupMachineVertex(
                child_vertices, max_n_atoms, is_final_row)
            vertex.add_constraint(EarConstraint())

        globals_variables.get_simulator().add_machine_vertex(vertex)
        if len(mv_edges) > 0:
            for (j, partition_name) in mv_edges:
                if j > vertex_slice.lo_atom:  # vertex not already built
                    # add to the "to build" edge list
                    self._todo_edges.append(
                        (vertex_slice.lo_atom, j, partition_name))
                else:
                    # add edge instance
                    try:
                        globals_variables.get_simulator().add_machine_edge(
                            MachineEdge(
                                vertex, self._mv_list[j], label="spinnakear"),
                            partition_name)
                    except IndexError:
                        print

        self._mv_list.append(vertex)

        if vertex_slice.lo_atom == self._n_atoms -1:
            # all vertices have been generated so add all incomplete edges
            for (source, target, partition_name) in self._todo_edges:
                globals_variables.get_simulator().add_machine_edge(
                    MachineEdge(
                        self._mv_list[source], self._mv_list[target],
                        label="spinnakear"),
                    partition_name)
        return vertex

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_atoms

    def _calculate_n_atoms(self, n_group_tree_rows):

        # list indices correspond to atom index
        mv_index_list = []

        # each entry is a list of tuples containing mv indices the mv connects
        # to and the data partition name for the edge
        edge_index_list = []

        # each entry is the ID of the parent vertex - used to obtain parent
        # spike IDs
        parent_index_list = []
        ome_indices = []
        ome_index = len(mv_index_list)
        ome_indices.append(ome_index)
        mv_index_list.append('ome')
        parent_index_list.append([])
        edge_index_list.append([])

        for _ in range(self._model.n_channels):
            drnl_index = len(mv_index_list)
            mv_index_list.append('drnl')
            parent_index_list.append([ome_index])
            edge_index_list.append([])

            # Add the data edges (OME->DRNLs) to the ome entry in the edge list
            edge_index_list[ome_index].append(
                (drnl_index, data_partition_dict['ome']))
            fibres = []
            for __ in range(self._model.n_hsr_per_ihc):
                fibres.append(2)
            for ___ in range(self._model.n_msr_per_ihc):
                fibres.append(1)
            for ____ in range(self._model.n_lsr_per_ihc):
                fibres.append(0)

            random.shuffle(fibres)

            for j in range(self._model.n_ihcs):
                ihc_index = len(mv_index_list)

                # randomly pick fibre types
                chosen_indices = [
                    fibres.pop() for _ in range(self._model.n_fibres_per_ihcan)]

                mv_index_list.append(
                    'ihc{}{}{}'.format(
                        chosen_indices.count(0), chosen_indices.count(1),
                        chosen_indices.count(2)))

                # drnl data/command
                # add the IHC mv index to the DRNL edge list entries
                edge_index_list[drnl_index].append(
                    (ihc_index, data_partition_dict['drnl']))
                # add the drnl parent index to the ihc
                parent_index_list.append([drnl_index])
                edge_index_list.append([])

        # generate ihc seeds
        n_ihcans = self._model.n_channels * self._model.n_ihcs
        random_range = numpy.arange(n_ihcans * 4, dtype=numpy.uint32)
        ihc_seeds = numpy.random.choice(
            random_range, int(n_ihcans * 4), replace=False)

        # now add on the AN Group vertices
        # builds the binary tree aggregator
        n_child_per_group = self._MAX_N_ATOMS_PER_CORE
        n_angs = n_ihcans

        for row_index in range(n_group_tree_rows):
            n_row_angs = int(numpy.ceil(float(n_angs) / n_child_per_group))
            if row_index > 0:
                ang_indices = [
                    i for i, label in enumerate(mv_index_list)
                    if label == "inter_{}".format(row_index-1)]
            else:
                ang_indices = [
                    i for i, label in enumerate(mv_index_list)
                    if "ihc" in label]

            for an in range(n_row_angs):
                if row_index == n_group_tree_rows-1:
                    mv_index_list.append("group_{}".format(row_index))
                else:
                    mv_index_list.append("inter_{}".format(row_index))
                edge_index_list.append([])
                ang_index = len(mv_index_list) - 1

                # find child ihcans
                child_indices = ang_indices[
                    an * n_child_per_group:an * n_child_per_group +
                    n_child_per_group]
                parent_index_list.append(child_indices)
                for i in child_indices:
                    edge_index_list[i].append((ang_index, 'AN'))
            n_angs = n_row_angs

        return (len(mv_index_list), mv_index_list, parent_index_list,
                edge_index_list, ihc_seeds, ome_indices)

    @overrides(HandOverToVertex.source_vertices_from_edge)
    def source_vertices_from_edge(self, projection):
        """ returns vertices for connecting this projection

        :param projection: projection to connect to sources
        :return: the iterable of vertices to be sources of this projection
        """

    @overrides(HandOverToVertex.destination_vertices_from_edge)
    def destination_vertices_from_edge(self, projection):
        """ return vertices for connecting this projection

        :param projection: projection to connect to destinations
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
