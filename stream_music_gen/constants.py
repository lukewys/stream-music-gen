import os

# Tokenization parameters
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

# Encodec model constants
ENCODEC_MODEL_NAME = "facebook/encodec_32khz"
ENCODEC_SAMPLE_RATE = 32000
ENCODEC_NUM_CODEBOOK_PER_LAYER = 2048
ENCODEC_FRAME_RATE_HZ = 50
ENCODEC_EMB_DIM = 128

# DAC model constants
DAC_SAMPLE_RATE = 32000
DAC_NUM_CODEBOOK_PER_LAYER = 4096
DAC_FRAME_RATE_HZ = 50
DAC_EMB_DIM = 1024
# DAC_PRETRAINED_MODEL_PATH: the "pretrained_models/250121_stemmix_dac_weights_400k_steps.pth" relative to the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAC_PRETRAINED_MODEL_PATH = os.path.join(
    root_dir, "pretrained_models/250121_stemmix_dac_weights_400k_steps.pth"
)

RMS_FRAME_RATE_HZ = 50


# Evaluation Sample Rates
EVAL_SAMPLE_RATE = 16000


# Dataset splits mapping
DATASET_SPLITS = {
    "slakh2100": {
        "train": "train",
        "valid": "validation",
        "test": "test",
    },
    "cocochorales": {
        "train": "train",
        "valid": "valid",
        "test": "test",
    },
    "moisesdb": {
        "train": "train",
        "valid": "test",
        "test": "test",
    },
    "musdb": {
        "train": "train",
        "valid": "test",
        "test": "test",
    },
}


# MIDI related constants
MIDI_CATEGORIES_TO_MIDI_PROGRAM = {
    "Piano": range(1, 9),
    "Chromatic Percussion": range(9, 17),
    "Organ": range(17, 25),
    "Guitar": range(25, 33),
    "Bass": range(33, 41),
    "Strings": range(41, 49),
    "Ensemble": range(49, 57),
    "Brass": range(57, 65),
    "Reed": range(65, 73),
    "Pipe": range(73, 81),
    "Synth Lead": range(81, 89),
    "Synth Pad": range(89, 97),
    "Synth Effects": range(97, 105),
    "Ethnic": range(105, 113),
    "Percussive": range(113, 121),
    "Sound Effects": range(121, 129),
}

# Synonyms are included in the list
MIDI_CATEGORIES = [
    [
        "Piano",
        "other_sounds_(hapischord,_melotron_etc)",
    ],
    ["Chromatic Percussion"],
    [
        "Organ",
        "other_wind",  # in moises db, other_wind is harmonica
    ],
    ["Guitar"],
    ["Bass"],
    [
        "Strings",
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Double Bass",
        "string_section",
    ],
    ["Ensemble"],
    ["Brass"],
    ["Reed"],
    ["Pipe"],
    ["Synth Lead", "synth_lead"],
    ["Synth Pad", "synth_pad"],
    ["Synth Effects"],
    ["Ethnic", "banjo,_mandolin,_ukulele,_harp_etc"],
    [
        "Percussive",
        "Drums",
        "Drum",
        "cymbals",
        "hihat",
        "snare",
        "kick",
        "tom",
        "hi_hat",
        "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)",
        "pitched_percussion_(mallets,_glockenspiel,_...)",
        "overheads",
    ],
    [
        "Sound Effects",
        "fx/processed_sound,_scratches,_gun_shots,_explosions_etc",
    ],
    [
        "Vocal",
        "Vocals",
        "Voice",
        "lead_male_singer",
        "lead_female_singer",
        "other_(vocoder,_beatboxing_etc)",
    ],
    [
        "Unknown",
        "other",  # in musedb, other is the mix of all other instruments
    ],
]


# case unsensitive
MIDI_CATEGORIES = [
    [c.lower() for c in category] for category in MIDI_CATEGORIES
]


def inst_name_to_inst_class_id(inst_name: str) -> int:
    inst_name = inst_name.lower()
    for i, category in enumerate(MIDI_CATEGORIES):
        if any([c in inst_name for c in category]):
            return i
    print(f"Unknown instrument name: {inst_name}")
    return len(MIDI_CATEGORIES) - 1  # unknown


def midi_prog_to_inst_class_id(midi_program: int) -> int:
    for category, program_range in MIDI_CATEGORIES_TO_MIDI_PROGRAM.items():
        if midi_program in program_range:
            return inst_name_to_inst_class_id(category)
    print(f"Unknown midi program: {midi_program}")
    return inst_name_to_inst_class_id("Unknown")
