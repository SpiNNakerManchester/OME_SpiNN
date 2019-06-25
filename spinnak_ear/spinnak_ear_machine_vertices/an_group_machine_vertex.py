from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
# from pacman.model.resources.sdram_resource import SDRAMResource
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
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from enum import Enum
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.simulation import simulation_utilities

from data_specification.constants import APP_PTR_TABLE_BYTE_SIZE

import numpy

from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
import numpy as np


class ANGroupMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition):
    """ A vertex that runs the multi-cast acknowledge algorithm
    """

    AN_GROUP_PARTITION_IDENTIFER = "AN"

    # The data type of the keys
    _KEY_ELEMENT_TYPE = DataType.UINT32
    _KEY_MASK_ENTRY_DTYPE = [
        ("key", "<u4"), ("mask", "<u4"),("offset", "<u4")]
    _KEY_MASK_ENTRY_SIZE_BYTES = 12
    _N_PARAMETER_BYTES = 5 * 4

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    def __init__(self, n_atoms, is_final_row):
        """
        """
        MachineVertex.__init__(self, label="AN Group Node", constraints=None)
        self._n_atoms = n_atoms
        self._is_final_row = is_final_row

    def add_child_vertex(self, child):
        self._child_vertices.append(child)

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        # sdram = self._N_PARAMETER_BYTES + len(self._child_vertices) * self._KEY_MASK_ENTRY_SIZE_BYTES
        sdram = self._N_PARAMETER_BYTES + 2 * self._KEY_MASK_ENTRY_SIZE_BYTES
        sdram += constants.SYSTEM_BYTES_REQUIREMENT + 8
        sdram += APP_PTR_TABLE_BYTE_SIZE

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
            time_scale_factor, routing_info, tags, placements,machine_graph):

        # Setup words + 1 for flags + 1 for recording size
        setup_size = constants.SYSTEM_BYTES_REQUIREMENT + 8
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=setup_size, label='systemInfo')

        # Reserve and write the parameters region
        region_size = (
            self._N_PARAMETER_BYTES + len(self._child_vertices) *
            self._KEY_MASK_ENTRY_SIZE_BYTES)
        spec.reserve_memory_region(self.REGIONS.PARAMETERS.value, region_size)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the number of child nodes
        spec.write_value(
            len(self._child_vertices), data_type=self._KEY_ELEMENT_TYPE)

        # Write the routing key
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)
        if len(partitions) == 0:
            # write 0 key
            spec.write_value(0)
            # write false is_key
            spec.write_value(0)
        for partition in partitions:
            if partition.identifier == 'SPIKE' or partition.identifier == 'AN':
                rinfo = routing_info.get_routing_info_from_partition(
                    partition)
                key = rinfo.first_key
                spec.write_value(
                   key, data_type=self._KEY_ELEMENT_TYPE)
                # write true is_key
                spec.write_value(1)
                break
        # write is final
        spec.write_value(self._is_final_row,data_type=self._KEY_ELEMENT_TYPE)
        # write n_atoms
        spec.write_value(self._n_atoms,data_type=self._KEY_ELEMENT_TYPE)

        # key and mask table generation
        key_and_mask_table = numpy.zeros(
            len(self._child_vertices), dtype=self._KEY_MASK_ENTRY_DTYPE)
        offset = 0
        for i, vertex in enumerate(self._child_vertices):
            key_and_mask = \
                routing_info.get_routing_info_from_pre_vertex(
                    vertex,'AN').first_key_and_mask
            key_and_mask_table[i]['key']=key_and_mask.key
            key_and_mask_table[i]['mask']=key_and_mask.mask
            key_and_mask_table[i]['offset']=offset
            offset += vertex._n_atoms

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