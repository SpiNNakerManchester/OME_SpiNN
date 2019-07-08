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
from spinn_front_end_common.utilities.utility_objs import ExecutableType
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
        AbstractProvidesNKeysForPartition):
    """ A vertex that runs the multi-cast acknowledge algorithm
    """

    AN_GROUP_PARTITION_IDENTIFER = "AN"

    # The data type of the keys
    _KEY_MASK_ENTRY_DTYPE = [
        ("key", "<u4"), ("mask", "<u4"),("offset", "<u4")]

    _KEY_MASK_ENTRY_SIZE_BYTES = 12

    # 1 n child, 2. has key, 3. key, 4. is final, 5= n atoms
    _N_PARAMETER_BYTES = 5 * constants.WORD_TO_BYTE_MULTIPLIER

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1)])

    def __init__(self, n_atoms, n_children, is_final_row, final_row_lo_atom):
        """
        """
        MachineVertex.__init__(
            self,
            label="AN Group Node with lo atom {} and is {} for final "
                  "row".format(final_row_lo_atom, is_final_row),
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

    @inject_items({
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
        "routing_info": "MemoryRoutingInfos",
        "tags": "MemoryTags",
        "placements": "MemoryPlacements",
        "machine_graph":"MemoryMachineGraph",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "machine_time_step", "time_scale_factor","routing_info", "tags",
            "placements","machine_graph"])
    def generate_data_specification(
            self, spec, placement, machine_time_step,
            time_scale_factor, routing_info, tags, placements, machine_graph):

        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve and write the parameters region
        region_size = (
            self._N_PARAMETER_BYTES +
            (self._n_children * self._KEY_MASK_ENTRY_SIZE_BYTES))
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(
            simulation_utilities.get_simulation_header_array(
                self.get_binary_file_name(), machine_time_step,
                time_scale_factor))

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the number of child nodes
        spec.write_value(self.n_atoms)

        # Write the routing key
        partitions = list(
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(
                self))
        if len(partitions) == 0:
            # write 0 key
            spec.write_value(0)
            # write false is_key
            spec.write_value(0)
        else:
            key = routing_info.get_first_key_from_partition(partitions[0])
            spec.write_value(key)
            # write true is_key
            spec.write_value(1)

        # write is final
        spec.write_value(self._is_final_row)
        # write n_atoms
        spec.write_value(self._n_atoms)

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

        # End the specification
        spec.end_specification()

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_atoms  # two for control IDs

    @property
    def n_atoms(self):
        return self._n_atoms
