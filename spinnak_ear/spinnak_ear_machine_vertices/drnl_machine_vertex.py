from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
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
from spinn_front_end_common.interface.buffer_management.buffer_models\
    .abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.utilities.constants import BYTES_PER_WORS
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.interface.profiling.profile_data \
    import ProfileData
from enum import Enum
import numpy

from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

from spinn_front_end_common.interface.profiling.abstract_has_profile_data \
    import AbstractHasProfileData
from spinn_front_end_common.interface.profiling import profile_utils
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinnak_ear.spinnak_ear_machine_vertices.abstract_ear_profiled import \
    AbstractEarProfiled
from spinnak_ear.spinnak_ear_machine_vertices.ome_machine_vertex import \
    OMEMachineVertex


class DRNLMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        AbstractProvidesNKeysForPartition,
        AbstractEarProfiled,
        AbstractReceiveBuffersToHost):

    """ A vertex that runs the DRNL algorithm
    """

    __slots__ = [
        "_cf",
        "_fs",
        "_delay",
        "_drnl_index",
        "_is_recording",
        "_moc_vertices",
        "_num_data_points",
        "_n_moc_data_points",
        "_recording_size",
        "_profile",
        "_n_profile_samples",
        "_process_profile_times",
        "_filter_params"
    ]

    # the outgoing partition id for DRNL
    DRNL_PARTITION_ID = "DRNLData"

    # The number of bytes for the parameters
    # 1: n data points, 2: ome core, 3. my core, 4: ome app id?? 5: ack key?
    # 6: data key, 7: n ihcans 8: centre freq, 9: delay 10: sampling freq,
    # 11: ome data key, 12: recording flag, 13: n moc mvs, 14: conn lut size
    _N_PARAMETER_BYTES = 14 * BYTES_PER_WORS

    MOC_RECORDING_REGION_ID = 0

    REGIONS = Enum(
        value="REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMETERS', 1),
               ('RECORDING', 2),
               ('PROFILE', 3)])

    def __init__(
            self, cf, delay, fs, n_data_points, drnl_index, is_recording,
            profile):
        """

        :param ome: The connected ome vertex
        """
        MachineVertex.__init__(
            self, label="DRNL Node of {}".format(drnl_index), constraints=None)
        AbstractEarProfiled.__init__(self, profile, self.REGIONS.Profile.value)
        AbstractProvidesNKeysForPartition.__init__(self)

        self._cf = cf
        self._fs = fs
        self._delay = int(delay)
        self._drnl_index = drnl_index
        self._is_recording = is_recording

        self._moc_vertices = list()

        self._num_data_points = n_data_points
        self._n_moc_data_points = int(
            (self._num_data_points / (self._fs / 1000.0)) / 10) * 10

        # recording size
        self._recording_size = (
            self._n_moc_data_points * DataType.FLOAT_64.size +
            DataType.UINT32.size)

        # filter params
        self._filter_params = self._calculate_filter_parameters()

    def _get_data_key(self, routing_info):
        key = routing_info.get_first_key_from_pre_vertex(
            self, self.DRNL_PARTITION_ID)
        if key is None:
            raise Exception("no drnl key generated!")
        return key

    def _calculate_filter_parameters(self):
        """
        
        :return: 
        """
        dt = 1./self._fs
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
    def n_data_points(self):
        return self._num_data_points

    def _param_region_size(self):
        # param region
        sdram = self._N_PARAMETER_BYTES
        sdram += len(self._filter_params) * DataType.FLOAT_64.size
        return sdram

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        # system region
        sdram = constants.SYSTEM_BYTES_REQUIREMENT
        # param region
        sdram += self._param_region_size()
        sdram += self._profile_size()
        sdram += self._recording_size

        #TODO i think this is the sdram links
        sdram += 4 * 8 * 4 #circular buffer to IHCs

        resources = ResourceContainer(
            dtcm=DTCMResource(0),
            sdram=ConstantSDRAM(sdram),
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
    def get_n_keys_for_partition(self, _, __):
        return 1

    def _reserve_memory_regions(self, spec):
        # Setup words + 1 for flags + 1 for recording size
        # reserve system region
        spec.reserve_memory_region(
            region=self.REGIONS.SYSTEM.value,
            size=constants.SIMULATION_N_BYTES, label='systemInfo')

        # Reserve the parameters region
        spec.reserve_memory_region(
            self.REGIONS.PARAMETERS.value, self._param_region_size())

        # reserve recording region
        spec.reserve_memory_region(
            self.REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))

        # handle profile stuff
        self._reserve_profile_memory_regions(spec)

    def _write_param_region(
            self, spec, machine_graph, placement, placements, routing_info):
        spec.switch_write_focus(self.REGIONS.PARAMETERS.value)

        # Write the data size in words
        spec.write_value(self._num_data_points)

        ome_processor = None
        ome_data_key = None
        for edge in machine_graph.get_edges_ending_at_vertex(self):
            if isinstance(edge.pre_vertex, OMEMachineVertex):
                ome_processor = (
                    placements.get_placement_of_vertex(edge.pre_vertex).p)
                ome_data_key = routing_info.get_first_key_for_edge(edge)

        # Write the OMECoreID
        spec.write_value(ome_processor)

        # Write the CoreID
        spec.write_value(placement.p)

        # Write the OMEAppID #TODO delete
        spec.write_value(0)

        # Write the Acknowledge key #TODO delete
        spec.write_value(0)

        # Write the key
        spec.write_value(self._get_data_key(routing_info))

        # Write number of ihcans
        spec.write_value(len(machine_graph.get_edges_starting_at_vertex(self)))

        # Write the centre frequency
        spec.write_value(self._cf)

        # Write the delay
        spec.write_value(self._delay)

        # Write the sampling frequency
        spec.write_value(self._fs)

        # write the OME data key
        spec.write_value(ome_data_key)

        # write is recording
        spec.write_value(int(self._is_recording))

        # write the filter params
        for param in self._filter_params:
            spec.write_value(param, data_type=DataType.FLOAT_64)

        # Write the number of mocs
        spec.write_value(0)
        # Write the size of the conn LUT
        spec.write_value(0)

    @inject_items({
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
        "machine_graph": "MemoryMachineGraph",
        "routing_info": "MemoryRoutingInfos",
        "placements": "MemoryPlacements",
        "tags": "MemoryTags",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=[
            "machine_time_step", "time_scale_factor", "machine_graph",
            "routing_info", "placements", "tags"])
    def generate_data_specification(
            self, spec, placement, machine_time_step, time_scale_factor,
            machine_graph, routing_info, placements, tags):

        # reserve regions
        self._reserve_memory_regions(spec)

        # simulation.c requirements
        spec.switch_write_focus(self.REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # params
        self._write_param_region(
            spec, machine_graph, placement, placements, routing_info)

        # only write params if used
        self._write_profile_dsg(spec)

        # Write the recording regions
        spec.switch_write_focus(self.REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # End the specification
        spec.end_specification()

    def read_moc_attenuation(self, buffer_manager, placement):
        """
         Read back the spikes 
        :param buffer_manager: buffer manager
        :param placement: placement
        :return: output data
        """

        data, _ = buffer_manager.get_data_by_placement(
            placement, self.MOC_RECORDING_REGION_ID)
        formatted_data = (
            numpy.array(data, dtype=numpy.uint8, copy=True).view(numpy.float64))
        output_data = formatted_data.copy()
        output_length = len(output_data)

        # check all expected data has been recorded
        if output_length != self._n_moc_data_points:
            # if not set output to zeros of correct length, this will cause
            # an error flag in run_ear.py raise Warning
            print(
                "recording not complete, reduce Fs or disable RT!\n"
                "recorded output length:{}, expected length:{} "
                "at placement:{},{},{}".format(
                    len(output_data), self._n_moc_data_points, placement.x,
                    placement.y, placement.p))

            output_data.resize(self._n_moc_data_points, refcheck=False)
        return output_data

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        if self._is_recording:
            regions = [self.MOC_RECORDING_REGION_ID]
        else:
            regions = []
        return regions

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, transciever, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self.REGIONS.RECORDING.value, transciever)
