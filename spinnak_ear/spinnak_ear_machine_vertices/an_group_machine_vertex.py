from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources import ConstantSDRAM
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource
from pacman.model.decorators.overrides import overrides
from pacman.executor.injection_decorator import inject_items

from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.impl.\
    machine_supports_auto_pause_and_resume import \
    MachineSupportsAutoPauseAndResume
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.utilities.utility_objs import ExecutableType, \
    ProvenanceDataItem
from spinn_front_end_common.utilities import constants
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

import numpy
from enum import Enum


class ANGroupMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition, MachineSupportsAutoPauseAndResume,
        ProvidesProvenanceDataFromMachineImpl):
    """ A vertex that runs the multi-cast acknowledge algorithm
    """

    # provenance items
    EXTRA_PROVENANCE_DATA_ENTRIES = Enum(
        value="EXTRA_PROVENANCE_DATA_ENTRIES",
        names=[("N_SPIKES", 0),
               ("N_PROVENANCE_ELEMENTS", 1)])

    AN_GROUP_PARTITION_IDENTIFIER = "AN"

    # The data type of the keys
    _KEY_MASK_ENTRY_DTYPE = [
        ("key", "<u4"), ("mask", "<u4"), ("offset", "<u4")]

    _KEY_MASK_ENTRY_SIZE_BYTES = 12

    # 1 n child, 2. has key, 3. key, 4. is final, 5= n atoms
    _N_PARAMETER_BYTES = 5 * constants.WORD_TO_BYTE_MULTIPLIER

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('KEY_MAP', 2),
               ('PROVENANCE', 3)])

    def __init__(
            self, n_atoms, n_children, is_final_row, final_row_lo_atom, row):
        """
        """
        MachineVertex.__init__(
            self,
            label="AN Group Node with lo atom {} and is {} for final "
                  "row in row {}".format(final_row_lo_atom, is_final_row, row),
            constraints=None)
        self._n_atoms = n_atoms
        self._n_children = n_children
        self._is_final_row = is_final_row
        self._low_atom = final_row_lo_atom

    @property
    def is_final_row(self):
        return self._is_final_row

    @property
    def low_atom(self):
        return self._low_atom

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        sdram += self._N_PARAMETER_BYTES
        sdram += self._KEY_MASK_ENTRY_SIZE_BYTES * self._n_children
        # provenance region
        sdram += self.get_provenance_data_size(
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value)

        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=ConstantSDRAM(sdram),
            cpu_cycles=CPUCyclesPerTickResource(0),
            iptags=[], reverse_iptags=[])
        return resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "SpiNNakEar_ANGroup.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        # return ExecutableType.SYNC
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.REGIONS.PROVENANCE.value

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.EXTRA_PROVENANCE_DATA_ENTRIES.N_PROVENANCE_ELEMENTS.value

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        n_spikes = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_SPIKES.value]
        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "n spikes transmitted"), n_spikes))
        return provenance_items

    def _reserve_memory_regions(self, spec):
        """ reserve dsg regions

        :param spec: dsg spec
        :rtype: None
        """
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value, self._N_PARAMETER_BYTES)

        # Reserve the keys map region
        spec.reserve_memory_region(
            self.REGIONS.KEY_MAP.value,
            self._n_children * self._KEY_MASK_ENTRY_SIZE_BYTES)

        # reserve provenance data region
        self.reserve_provenance_data_region(spec)

    def _fill_in_params_region(self, spec, machine_graph, routing_info):
        """ fills in the dsg region for params

        :param spec: dsg spec
        :param machine_graph: machine graph
        :param routing_info: routing info
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the number of child nodes
        spec.write_value(self._n_children)

        # Write the routing key
        partitions = list(
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(
                self))
        if len(partitions) == 0:
            # write false is_key
            spec.write_value(0)
            # write 0 key
            spec.write_value(0)
        else:
            # write true is_key
            spec.write_value(1)

            key = routing_info.get_first_key_from_partition(partitions[0])
            spec.write_value(key)

        # write is final
        spec.write_value(self._is_final_row)
        # write n_atoms
        spec.write_value(self._n_atoms)

    def _fill_in_key_map_region(self, spec, machine_graph, routing_info):
        """ fill in the key map region

        :param spec: dsg spec
        :param machine_graph: machine graph
        :param routing_info: routing info
        :rtype: None
        """
        spec.switch_write_focus(self.REGIONS.KEY_MAP.value)

        # key and mask table generation
        key_and_mask_table = numpy.zeros(
            self._n_children, dtype=self._KEY_MASK_ENTRY_DTYPE)

        # build master pop table thing
        offset = 0
        for i, incoming_edge in enumerate(
                machine_graph.get_edges_ending_at_vertex(self)):
            key_and_mask = routing_info.get_routing_info_for_edge(
                incoming_edge).first_key_and_mask
            key_and_mask_table[i]['key'] = key_and_mask.key
            key_and_mask_table[i]['mask'] = key_and_mask.mask
            key_and_mask_table[i]['offset'] = offset
            offset += key_and_mask.n_keys

        # sort entries by key
        key_and_mask_table.sort(order='key')
        spec.write_array(key_and_mask_table.view("<u4"))

    @inject_items({
        "time_period_map": "MachineTimeStepMap",
        "time_scale_factor": "TimeScaleFactor",
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "machine_graph": "MemoryMachineGraph",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "time_period_map", "time_scale_factor", "routing_info", "tags",
            "placements", "machine_graph"])
    def generate_data_specification(
            self, spec, placement, time_period_map,
            time_scale_factor, routing_info, tags, placements, machine_graph):

        # reserve regions
        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(
            simulation_utilities.get_simulation_header_array(
                self.get_binary_file_name(), time_period_map[self],
                time_scale_factor))

        # app level regions fill in
        self._fill_in_params_region(spec, machine_graph, routing_info)
        self._fill_in_key_map_region(spec, machine_graph, routing_info)

        # End the specification
        spec.end_specification()

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_atoms  # two for control IDs

    @property
    def n_atoms(self):
        return self._n_atoms
