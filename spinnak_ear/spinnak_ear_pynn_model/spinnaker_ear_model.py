import os

import numpy as np

from spinn_front_end_common.utilities import globals_variables
from spinn_utilities.overrides import overrides
from spinnak_ear import model_binaries
from spinnak_ear.spinnak_ear_application_vertex.spinnakear_application_vertex \
    import SpiNNakEarApplicationVertex
from spynnaker.pyNN.models.abstract_pynn_model import AbstractPyNNModel


class SpiNNakEar(AbstractPyNNModel):

    # no extra params
    default_population_parameters = {}

    # defaults magic numbers
    _DEFAULT_MAX_INPUT_TO_AGGREGATION_GROUP = 2
    _DEFAULT_N_LSR_PER_IHC = 2
    _DEFAULT_N_MSR_PER_IHC = 2
    _DEFAULT_N_HSR_PER_IHC = 6
    _DEFAULT_AUDIO_SAMPLING_FREQUENCY = 22050.0
    _DEFAULT_RANDOM_SEED = 44444
    _DEFAULT_RESAMPLE_FACTOR = 1
    _DEFAULT_SEG_SIZE = 8
    _DEFAULT_PROFILE = False
    _DEFAULT_N_BUFFERS_IN_SDRAM_TOTAL = 4

    # scale max
    FULL_SCALE = 1.0

    # audio segment size
    SEG_SIZE = 8

    # default params
    DEFAULT_PARAMS = {
        'audio_input': None,
        'fs': _DEFAULT_AUDIO_SAMPLING_FREQUENCY,
        'pole_freqs': None,
        'param_file': None,
        'ear_index': 0,
        # conflict with n neurons. needs thinking
        'scale': FULL_SCALE,
        'n_lsr_per_ihc': _DEFAULT_N_LSR_PER_IHC,
        'n_msr_per_ihc': _DEFAULT_N_MSR_PER_IHC,
        'n_hsr_per_ihc': _DEFAULT_N_HSR_PER_IHC,
        'min_audio_frequency':
            SpiNNakEarApplicationVertex.DEFAULT_MIN_AUDIO_FREQUENCY,
        'max_audio_frequency':
            SpiNNakEarApplicationVertex.DEFAULT_MAX_AUDIO_FREQUENCY,
        'ihcan_fibre_random_seed': _DEFAULT_RANDOM_SEED,
        'profile': _DEFAULT_PROFILE,
        'ihc_seeds_seed': _DEFAULT_RANDOM_SEED,
        'max_input_to_aggregation_group':
            _DEFAULT_MAX_INPUT_TO_AGGREGATION_GROUP,
        'resample_factor': _DEFAULT_RESAMPLE_FACTOR,
        # auto generate from thesis (robert James's)
        'seq_size': _DEFAULT_SEG_SIZE,
        "n_buffers_in_sdram_total": _DEFAULT_N_BUFFERS_IN_SDRAM_TOTAL,
    }

    NAME = "SpikeSourceSpiNNakEar"

    __slots__ = [
        #
        "_audio_input",
        #
        "_fs",
        #
        '_pole_freqs',
        #
        "_param_file",
        # left or right ear
        "_ear_index",
        # scale between all ear and mini versions
        "_scale",
        # human readable name of model
        "_model_name",
        # how many cores for ihc
        "_n_fibres_per_ihc",
        # how many lsrs in each ihcan
        '_n_lsr_per_ihc',
        # how many msr per ihc
        '_n_msr_per_ihc',
        # how many hsr per ihc
        '_n_hsr_per_ihc',
        # min audio freq
        '_min_audio_frequency',
        # max audio freq
        '_max_audio_frequency',
        #
        "_ihcan_fibre_random_seed",
        #
        "_profile",
        #
        "_ihc_seeds_seed",
        # number of atoms/ keys to allow each aggregation node to process
        "_max_input_to_aggregation_group",
        # resample factor
        "_resample_factor",
        #
        "_seq_size",
        #
        "_n_buffers_in_sdram_total",
        #
        "_app_vertex"
    ]

    def __init__(
            self, audio_input=DEFAULT_PARAMS['audio_input'],
            fs=DEFAULT_PARAMS['fs'],
            pole_freqs=DEFAULT_PARAMS['pole_freqs'],
            param_file=DEFAULT_PARAMS['param_file'],
            ear_index=DEFAULT_PARAMS['ear_index'],
            scale=DEFAULT_PARAMS['scale'],
            n_lsr_per_ihc=DEFAULT_PARAMS['n_lsr_per_ihc'],
            n_msr_per_ihc=DEFAULT_PARAMS['n_msr_per_ihc'],
            n_hsr_per_ihc=DEFAULT_PARAMS['n_hsr_per_ihc'],
            min_audio_frequency=DEFAULT_PARAMS['min_audio_frequency'],
            max_audio_frequency=DEFAULT_PARAMS['max_audio_frequency'],
            ihcan_fibre_random_seed=DEFAULT_PARAMS['ihcan_fibre_random_seed'],
            profile=DEFAULT_PARAMS['profile'],
            ihc_seeds_seed=DEFAULT_PARAMS['ihc_seeds_seed'],
            max_input_to_aggregation_group=DEFAULT_PARAMS[
                'max_input_to_aggregation_group'],
            resample_factor=DEFAULT_PARAMS['resample_factor'],
            seq_size=DEFAULT_PARAMS['seq_size'],
            n_buffers_in_sdram_total=DEFAULT_PARAMS[
                'n_buffers_in_sdram_total']):
        self._fs = fs
        self._pole_freqs = pole_freqs
        self._param_file = param_file
        self._ear_index = ear_index
        self._scale = scale
        self._model_name = self.NAME
        self._n_lsr_per_ihc = n_lsr_per_ihc
        self._n_msr_per_ihc = n_msr_per_ihc
        self._n_hsr_per_ihc = n_hsr_per_ihc
        self._n_fibres_per_ihc = sum(
            [n_lsr_per_ihc, n_msr_per_ihc, n_hsr_per_ihc])
        self._min_audio_frequency = min_audio_frequency
        self._max_audio_frequency = max_audio_frequency
        self._ihcan_fibre_random_seed = ihcan_fibre_random_seed
        self._ihc_seeds_seed = ihc_seeds_seed
        self._profile = profile
        self._max_input_to_aggregation_group = max_input_to_aggregation_group
        self._resample_factor = resample_factor
        self._seq_size = seq_size
        self._n_buffers_in_sdram_total = n_buffers_in_sdram_total
        self._app_vertex = None

        if self._seq_size == 0:
            raise Exception("The seq size must be greater than 0")

        if isinstance(audio_input, list):
            audio_input = np.asarray(audio_input)

        if len(audio_input.shape) > 1:
            raise Exception(
                "For binaural simulation please create separate "
                "SpiNNak-Ear populations (left/right)")

        if audio_input is None:
            audio_input = np.asarray([])

        self._audio_input = (
            audio_input[0:int(
                np.floor(len(audio_input) / self.SEG_SIZE) * self.SEG_SIZE)])
        self._audio_input = np.asarray(self._audio_input)

        # update finder to look inside ear model binaries location
        globals_variables.get_simulator().executable_finder.add_path(
            os.path.dirname(model_binaries.__file__))

    @overrides(AbstractPyNNModel.create_vertex)
    def create_vertex(self, n_neurons, label, constraints):
        self._app_vertex = SpiNNakEarApplicationVertex(
            n_neurons,  constraints, label, self, self._profile,
            globals_variables.get_simulator().time_scale_factor)
        return self._app_vertex

    def calculate_n_atoms(self):

        # figure how many hair bits per ihcan core
        n_fibres_per_ihcan_core = \
            SpiNNakEarApplicationVertex.fibres_per_ihcan_core(
                globals_variables.get_simulator().time_scale_factor / self._fs)

        # NOTE the wrapping to a whole number of channels
        n_channels = (
            (int((SpiNNakEarApplicationVertex.FULL_EAR_HAIR_FIBERS *
                  float(self._scale)) / self._n_fibres_per_ihc) *
             self._n_fibres_per_ihc) / self._n_fibres_per_ihc)

        # figure out how many atoms are aggregated during the aggregation tree
        atoms_per_row = SpiNNakEarApplicationVertex.calculate_atoms_per_row(
            n_channels, self._n_fibres_per_ihc, n_fibres_per_ihcan_core,
            self._max_input_to_aggregation_group)

        # figure out atoms
        atoms, _, _ = \
            SpiNNakEarApplicationVertex.calculate_n_atoms_for_each_vertex_type(
                atoms_per_row, self._max_input_to_aggregation_group,
                n_channels, self._n_fibres_per_ihc, self._seq_size)

        # return atoms
        return atoms

    @property
    def n_buffers_in_sdram_total(self):
        return self._n_buffers_in_sdram_total

    @property
    def seq_size(self):
        return self._seq_size

    @property
    def resample_factor(self):
        return self._resample_factor

    @property
    def max_input_to_aggregation_group(self):
        return self._max_input_to_aggregation_group

    @property
    def ihcan_fibre_random_seed(self):
        return self._ihcan_fibre_random_seed

    @property
    def model_name(self):
        return self._model_name

    @property
    def audio_input(self):
        return self._audio_input

    @property
    def profile(self):
        return self._profile

    @property
    def fs(self):
        return self._fs

    @property
    def pole_freqs(self):
        return self._pole_freqs

    @property
    def param_file(self):
        return self._param_file

    @property
    def ear_index(self):
        return self._ear_index

    @property
    def scale(self):
        return self._scale

    @property
    def n_fibres_per_ihc(self):
        return self._n_fibres_per_ihc

    @property
    def n_lsr_per_ihc(self):
        return self._n_lsr_per_ihc

    @property
    def n_msr_per_ihc(self):
        return self._n_msr_per_ihc

    @property
    def n_hsr_per_ihc(self):
        return self._n_hsr_per_ihc

    @property
    def min_audio_frequency(self):
        return self._min_audio_frequency

    @property
    def max_audio_frequency(self):
        return self._max_audio_frequency

    @property
    def ihc_seeds_seed(self):
        return self._ihc_seeds_seed
