from spinn_front_end_common.abstract_models import AbstractChangableAfterRun, \
    AbstractRewritesDataSpecification
from spynnaker.pyNN.models.abstract_models.\
    abstract_accepts_incoming_synapses import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractSpikeRecordable
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

from spinn_front_end_common.utilities import \
    globals_variables, helpful_functions
from spinn_front_end_common.utilities import \
    constants as front_end_common_constants

from data_specification.enums.data_type import DataType

from spinnak_ear_machine_vertices.ome_machine_vertex import OMEMachineVertex
from spinnak_ear_machine_vertices.drnl_machine_vertex import DRNLMachineVertex
from spinnak_ear_machine_vertices.ihcan_machine_vertex import \
    IHCANMachineVertex
from spinnak_ear_machine_vertices.an_group_machine_vertex import \
    ANGroupMachineVertex

import numpy as np
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
        AbstractRewritesDataSpecification, AbstractSpikeRecordable):

    __slots__ = [
        # pynn model
        '_model',
        # bool flag for neuron param changes
        '_change_requires_neuron_parameters_reload'
    ]

    FREQUENCY_ERROR = (
        "The input sampling frequency is too high for the chosen simulation " 
        "time scale. Please reduce Fs or increase the time scale factor in "
        "the config file")

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

    # n recording regions
    _N_POPULATION_RECORDING_REGIONS = 1

    _MAX_N_ATOMS_PER_CORE = 2
    _FINAL_ROW_N_ATOMS = 256
    _N_FIBRES_PER_IHCAN = 2
    _N_LSR_PER_IHC = 2
    _N_MSR_PER_IHC = 2
    _N_HSR_PER_IHC = 6
    _N_FIBRES_PER_IHC = _N_LSR_PER_IHC + _N_MSR_PER_IHC + _N_HSR_PER_IHC

    def __init__(
            self, n_neurons, constraints, label, model):
        # Superclasses
        ApplicationVertex.__init__(self, label, constraints)
        AbstractAcceptsIncomingSynapses.__init__(self)
        SimplePopulationSettable.__init__(self)
        HandOverToVertex.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractRewritesDataSpecification.__init__(self)
        AbstractSpikeRecordable.__init__(self)

        self._model = model
        self._change_requires_neuron_parameters_reload = True

        self._data_size_bytes = (
            (self._model.audio_input.size * self._DATA_ELEMENT_TYPE.size) +
            self._DATA_COUNT_TYPE.size)

        self._todo_edges = []

        n_group_tree_rows = int(np.ceil(math.log(
            (self._model.n_channels * self._N_FIBRES_PER_IHC) /
            self._N_FIBRES_PER_IHCAN, self._MAX_N_ATOMS_PER_CORE)))

        self._max_n_atoms_per_group_tree_row = (
            (self._MAX_N_ATOMS_PER_CORE ** np.arange(1, n_group_tree_rows+1)) *
            self._N_FIBRES_PER_IHCAN)

        self._max_n_atoms_per_group_tree_row = \
            self._max_n_atoms_per_group_tree_row[
                self._max_n_atoms_per_group_tree_row <=
                self._FINAL_ROW_N_ATOMS]

        self._n_group_tree_rows = self._max_n_atoms_per_group_tree_row.size

        self._is_recording_spikes = True#hack to force recording False
        self._is_recording_moc = True#hack to force recording False

        self._seed_index = 0
        self._pole_index = 0

        config = globals_variables.get_simulator().config
        self._time_scale_factor = helpful_functions.read_config_int(
            config, "Machine", "time_scale_factor")
        if self._model.fs / self._time_scale_factor > 22050:
            raise Exception(self.FREQUENCY_ERROR)

        if self._model.param_file is not None:
            try:
                pre_gen_vars = np.load(self._model.param_file)
                self._n_atoms = pre_gen_vars['n_atoms']
                self._mv_index_list = pre_gen_vars['mv_index_list']
                self._parent_index_list = pre_gen_vars['parent_index_list']
                self._edge_index_list = pre_gen_vars['edge_index_list']
                self._ihc_seeds = pre_gen_vars['ihc_seeds']
                self._ome_indices = pre_gen_vars['ome_indices']

            except:
                (self._n_atoms, self._mv_index_list, self._parent_index_list,
                 self._edge_index_list, self._ihc_seeds, self._ome_indices) = \
                    self.calculate_n_atoms(self._n_group_tree_rows)
                self.save_pre_gen_vars(self._model.param_file)

        self._size = n_neurons
        self._new_chip_indices = []
        drnl_count = 0
        for i, vertex_name in enumerate(self._mv_index_list):
            if vertex_name == "drnl":
                if drnl_count % 2 == 0:
                    self._new_chip_indices.append(i)
                drnl_count += 1

    def get_synapse_id_by_target(self, target):
        return 0

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        return 1 * machine_time_step

    def add_pre_run_connection_holder(
            self, connection_holder, projection_edge, synapse_information):
        super(SpiNNakEarApplicationVertex, self).add_pre_run_connection_holder(
            connection_holder, projection_edge, synapse_information)

    def save_pre_gen_vars(self, filepath):
        np.savez_compressed(
            filepath, n_atoms=self._n_atoms, mv_index_list=self._mv_index_list,
            parent_index_list=self._parent_index_list,
            edge_index_list=self._edge_index_list, ihc_seeds=self._ihc_seeds,
            ome_indices=self._ome_indices)

    def record(self, variables):
        if not isinstance(variables, list):
            variables = [variables]
        if len(variables) == 0:
            variables.append("all")
        for variable in variables:
            if variable == "spikes":
                self._is_recording_spikes = True
            elif variable == "moc":
                self._is_recording_moc = True
            elif variable == "all":
                self._is_recording_spikes = True
                self._is_recording_moc = True
            else:
                raise Exception(
                    "recording of " + variable +
                    " not supported by SpiNNak-Ear!")

    def get_data(self, variables):
        b_manager = globals_variables.get_simulator().buffer_manager
        output_data = {}
        if not isinstance(variables, list):
            variables = [variables]
        if len(variables) == 0:
            variables.append("all")
        if "all" in variables:
            variables = ['spikes', 'moc']
        for variable in variables:
            recorded_output = []
            drnl_indices = [
                i for i, label in enumerate(self._mv_index_list)
                if label == self.DRNL]
            progress = ProgressBar(
                len(drnl_indices),
                "reading ear {} ".format(self._model.ear_index) + variable)
            for drnl in drnl_indices:
                drnl_vertex = self._mv_list[drnl]
                channel_fibres = (drnl_vertex.read_samples(b_manager, variable))
                progress.update()
                for fibre in channel_fibres:
                    recorded_output.append(fibre)
            output_data[variable] = np.asarray(recorded_output)
            progress.end()
        return output_data

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, machine_time_step, using_extra_monitor_cores,
            placements=None, data_receiver=None,
            sender_extra_monitor_core_placement=None,
            extra_monitor_cores_for_router_timeout=None,
            handle_time_out_configuration=True, fixed_routes=None):

        super(SpiNNakEarApplicationVertex, self).get_connections_from_machine(
            transceiver, placement, edge, graph_mapper, routing_infos,
            synapse_information, machine_time_step, using_extra_monitor_cores,
            placements, data_receiver, sender_extra_monitor_core_placement,
            extra_monitor_cores_for_router_timeout,
            handle_time_out_configuration, fixed_routes)

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

        elif vertex_label == "drnl":
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
                    self._N_FIBRES_PER_IHCAN * np.ceil(
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
        self._change_requires_neuron_parameters_reload = True

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

    def calculate_n_atoms(self, n_group_tree_rows):

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
            for __ in range(self._N_HSR_PER_IHC):
                fibres.append(2)
            for ___ in range(self._N_MSR_PER_IHC):
                fibres.append(1)
            for ____ in range(self._N_LSR_PER_IHC):
                fibres.append(0)

            random.shuffle(fibres)

            for j in range(self._model.n_ihcs):
                ihc_index = len(mv_index_list)

                # randomly pick fibre types
                chosen_indices = [
                    fibres.pop() for _ in range(self._N_FIBRES_PER_IHCAN)]

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
        random_range = np.arange(n_ihcans * 4, dtype=np.uint32)
        ihc_seeds = np.random.choice(
            random_range, int(n_ihcans * 4), replace=False)

        # now add on the AN Group vertices
        n_child_per_group = self._MAX_N_ATOMS_PER_CORE
        n_angs = n_ihcans

        for row_index in range(n_group_tree_rows):
            n_row_angs = int(np.ceil(float(n_angs) / n_child_per_group))
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

    @overrides(AbstractRewritesDataSpecification.regenerate_data_specification)
    def regenerate_data_specification(self, spec, placement):
        """ Regenerate the data specification, only generating regions that\
            have changed and need to be reloaded
        """
        pass

    @overrides(
        AbstractRewritesDataSpecification.
        requires_memory_regions_to_be_reloaded)
    def requires_memory_regions_to_be_reloaded(self):
        """ Return true if any data region needs to be reloaded

        :rtype: bool
        """
        pass

    @overrides(AbstractRewritesDataSpecification.mark_regions_reloaded)
    def mark_regions_reloaded(self):
        """ Indicate that the regions have been reloaded
        """
        pass

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        pass

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        pass

    @overrides(AbstractSpikeRecordable.is_recording_spikes)
    def is_recording_spikes(self):
        pass

    @overrides(AbstractSpikeRecordable.get_spikes_sampling_interval)
    def get_spikes_sampling_interval(self):
        pass

    @overrides(AbstractSpikeRecordable.clear_spike_recording)
    def clear_spike_recording(self, buffer_manager, placements, graph_mapper):
        pass

    @overrides(AbstractSpikeRecordable.get_spikes)
    def get_spikes(
            self, placements, graph_mapper, buffer_manager, machine_time_step):
        pass

    @overrides(AbstractSpikeRecordable.set_recording_spikes)
    def set_recording_spikes(
            self, new_state=True, sampling_interval=None, indexes=None):
        pass
