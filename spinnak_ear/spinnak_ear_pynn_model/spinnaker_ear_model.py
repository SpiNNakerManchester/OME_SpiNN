from spynnaker.pyNN.models.abstract_pynn_model import AbstractPyNNModel
from spinn_utilities.overrides import overrides
from spinnak_ear.spinnak_ear_application_vertex.spinnakear_application_vertex\
    import SpiNNakEarApplicationVertex
import numpy as np


class SpiNNakEar(AbstractPyNNModel):

    # no extra params
    default_population_parameters = {}

    # defaults magic numbers
    _DEFAULT_N_FIBRES_PER_IHCAN = 2
    _DEFAULT_MAX_N_FIRES_PER_IHCAN = 2
    _DEFAULT_MAX_INPUT_TO_AGGREGATION_GROUP = 2
    _DEFAULT_N_LSR_PER_IHC = 2
    _DEFAULT_N_MSR_PER_IHC = 2
    _DEFAULT_N_HSR_PER_IHC = 6
    _DEFAULT_N_CHANNELS = 3000
    _DEFAULT_AUDIO_SAMPLING_FREQUENCY = 22050.0
    _DEFAULT_N_IHC_PER_DRNL = 5
    _DEFAULT_NYQUIST_RATIO = 0.5
    _DEFAULT_MAX_AUDIO_FREQUENCY = 17782
    _DEFAULT_MIN_AUDIO_FREQUENCY = 30
    _DEFAULT_RANDOM_SEED = 44444
    _DEFAULT_RESAMPLE_FACTOR = 1

    # scale max
    FULL_SCALE = 1.0

    # audio segment size
    SEG_SIZE = 8

    # biggest number of neurons for the ear model
    MAX_NEURON_SIZE = 30000.0

    # default params
    DEFAULT_PARAMS = {
        'audio_input': None,
        'fs': _DEFAULT_AUDIO_SAMPLING_FREQUENCY,
        'n_channels': _DEFAULT_N_CHANNELS,
        'pole_freqs': None,
        'param_file': None,
        'ear_index': 0,
        'scale': FULL_SCALE,
        'n_ihc': _DEFAULT_N_IHC_PER_DRNL,
        'n_fibres_per_ihcan': _DEFAULT_N_FIBRES_PER_IHCAN,
        'n_lsr_per_ihc': _DEFAULT_N_LSR_PER_IHC,
        'n_msr_per_ihc': _DEFAULT_N_MSR_PER_IHC,
        'n_hsr_per_ihc': _DEFAULT_N_HSR_PER_IHC,
        'nyquist_ratio': _DEFAULT_NYQUIST_RATIO,
        'min_audio_frequency': _DEFAULT_MIN_AUDIO_FREQUENCY,
        'max_audio_frequency': _DEFAULT_MAX_AUDIO_FREQUENCY,
        'ihcan_fibre_random_seed': _DEFAULT_RANDOM_SEED,
        'profile': False,
        'ihc_seeds_seed': _DEFAULT_RANDOM_SEED,
        'max_n_fibres_per_ihcan': _DEFAULT_MAX_N_FIRES_PER_IHCAN,
        'max_input_to_aggregation_group':
            _DEFAULT_MAX_INPUT_TO_AGGREGATION_GROUP,
        'resample_factor': _DEFAULT_RESAMPLE_FACTOR

    }

    NAME = "SpikeSourceSpiNNakEar"

    __slots__ = [
        #
        "_audio_input",
        #
        "_fs",
        #
        "_n_channels",
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
        "_n_ihc",
        # how many fibres in each ihcan
        '_n_fibres_per_ihcan',
        # how many lsrs in each ihcan
        '_n_lsr_per_ihc',
        # how many msr per ihc
        '_n_msr_per_ihc',
        # how many hsr per ihc
        '_n_hsr_per_ihc',
        # the nyquist ratio
        '_nyquist_ratio',
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
        #
        "_max_n_fibres_per_ihcan",
        # number of atoms/ keys to allow each aggregation node to process
        "_max_input_to_aggregation_group",
        # resample factor
        "_resample_factor"
    ]

    def __init__(
            self, audio_input=DEFAULT_PARAMS['audio_input'],
            fs=DEFAULT_PARAMS['fs'],
            n_channels=DEFAULT_PARAMS['n_channels'],
            pole_freqs=DEFAULT_PARAMS['pole_freqs'],
            param_file=DEFAULT_PARAMS['param_file'],
            ear_index=DEFAULT_PARAMS['ear_index'],
            scale=DEFAULT_PARAMS['scale'],
            n_ihc=DEFAULT_PARAMS['n_ihc'],
            n_fibres_per_ihcan=DEFAULT_PARAMS['n_fibres_per_ihcan'],
            n_lsr_per_ihc=DEFAULT_PARAMS['n_lsr_per_ihc'],
            n_msr_per_ihc=DEFAULT_PARAMS['n_msr_per_ihc'],
            n_hsr_per_ihc=DEFAULT_PARAMS['n_hsr_per_ihc'],
            nyquist_ratio=DEFAULT_PARAMS['nyquist_ratio'],
            min_audio_frequency=DEFAULT_PARAMS['min_audio_frequency'],
            max_audio_frequency=DEFAULT_PARAMS['max_audio_frequency'],
            ihcan_fibre_random_seed=DEFAULT_PARAMS['ihcan_fibre_random_seed'],
            profile=DEFAULT_PARAMS['profile'],
            ihc_seeds_seed=DEFAULT_PARAMS['ihc_seeds_seed'],
            max_n_fibres_per_ihcan=DEFAULT_PARAMS['max_n_fibres_per_ihcan'],
            max_input_to_aggregation_group=DEFAULT_PARAMS[
                'max_input_to_aggregation_group'],
            resample_factor=DEFAULT_PARAMS['resample_factor']):
        self._fs = fs
        self._n_channels = int(n_channels)
        self._pole_freqs = pole_freqs
        self._param_file = param_file
        self._ear_index = ear_index
        self._scale = scale
        self._model_name = self.NAME
        self._n_ihc = n_ihc
        self._n_fibres_per_ihcan = n_fibres_per_ihcan
        self._n_lsr_per_ihc = n_lsr_per_ihc
        self._n_msr_per_ihc = n_msr_per_ihc
        self._n_hsr_per_ihc = n_hsr_per_ihc
        self._nyquist_ratio = nyquist_ratio
        self._min_audio_frequency = min_audio_frequency
        self._max_audio_frequency = max_audio_frequency
        self._ihcan_fibre_random_seed = ihcan_fibre_random_seed
        self._ihc_seeds_seed = ihc_seeds_seed
        self._profile = profile
        self._max_n_fibres_per_ihcan = max_n_fibres_per_ihcan
        self._max_input_to_aggregation_group = max_input_to_aggregation_group
        self._resample_factor = resample_factor

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

        if pole_freqs is None:
            max_power = min(
                [np.log10(self._fs * self._nyquist_ratio),
                 np.log10(self._max_audio_frequency)])
            self._pole_freqs = np.flipud(np.logspace(
                np.log10(self._min_audio_frequency),
                max_power, self._n_channels))
        else:
            self._pole_freqs = pole_freqs

    @overrides(AbstractPyNNModel.create_vertex)
    def create_vertex(self, n_neurons, label, constraints):
        return SpiNNakEarApplicationVertex(
            n_neurons,  constraints, label, self, self._profile)

    @property
    def resample_factor(self):
        return self._resample_factor

    @property
    def max_n_fibres_per_ihcan(self):
        return self._max_n_fibres_per_ihcan

    @property
    def max_input_to_aggregation_group(self):
        return self._max_input_to_aggregation_group

    @property
    def ihcan_fibre_random_seed(self):
        return self._ihcan_fibre_random_seed

    @staticmethod
    def spinnakear_size_calculator(scale=FULL_SCALE):
        return int(SpiNNakEar.MAX_NEURON_SIZE * scale)

    @property
    def model_name(self):
        return self._model_name

    @property
    def audio_input(self):
        return self._audio_input

    @property
    def fs(self):
        return self._fs

    @property
    def n_channels(self):
        return self._n_channels

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
    def n_ihc(self):
        return self._n_ihc

    @property
    def n_fibres_per_ihcan(self):
        return self._n_fibres_per_ihcan

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
    def nyquist_ratio(self):
        return self._nyquist_ratio

    @property
    def min_audio_frequency(self):
        return self._min_audio_frequency

    @property
    def max_audio_frequency(self):
        return self._max_audio_frequency

    @property
    def ihc_seeds_seed(self):
        return self._ihc_seeds_seed

