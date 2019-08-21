from spinn_front_end_common.interface.profiling import \
    AbstractHasProfileData, profile_utils, ProfileData
from six import add_metaclass
from spinn_utilities.abstract_base import AbstractBase
from spinn_utilities.overrides import overrides


@add_metaclass(AbstractBase)
class AbstractEarProfiled(AbstractHasProfileData):

    PROFILER_N_SAMPLES = 10000

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    def __init__(self, profile, profile_region):
        # Set up for profiling
        self._profile = profile
        self._profile_region = profile_region
        self._n_profile_samples = self.PROFILER_N_SAMPLES
        self._process_profile_times = None

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        if self._profile:
            profiles = profile_utils.get_profiling_data(
                self._profile_region,
                self.PROFILE_TAG_LABELS, transceiver, placement)
            self._process_profile_times = profiles._tags['TIMER'][1]
        else:
            profiles = ProfileData(self.PROFILE_TAG_LABELS)
        return profiles

    def _reserve_profile_memory_regions(self, spec):
        # only reserve if using
        if self._profile:
            # reserve profile region
            profile_utils.reserve_profile_region(
                spec, self._profile_region, self._n_profile_samples)

    def _write_profile_dsg(self, spec):
        if self._profile:
            profile_utils.write_profile_region_data(
                spec, self._profile_region, self._n_profile_samples)

    def _profile_size(self):
        if self._profile:
            return profile_utils.get_profile_region_size(
                self._n_profile_samples)
        return 0
