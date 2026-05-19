# import tkinter as tk
# from tkinter import messagebox
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

mne.set_log_level("WARNING")


# ============================================================================
# DEFINICION DE FUNCIONES
# ============================================================================
def preprocessing_mne(
    path: str,
    file: str,
    excluded: list[str],
    bads: list[str],
    lowpass_cut: int,
    highpass_cut: int,
    raw_plot: bool,
    filtered_plot: bool,
    psd_plot: bool,
    edit_marks: bool,
    interpolate: bool = True,
):
    """Preprocessing of biosemi signals."""
    misc = ["EXG1", "EXG2"]
    eog = []
    filepath = path + file + ".fif"

    # Load raw data
    try:
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False, eog=eog, misc=misc, exclude=excluded)
    except:
        try:
            print("No se encontró el .bdf, probando con .fif")
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        except:
            print("No se pudo leer el archivo")

    print(raw.get_channel_types(picks=["EXG1", "EXG2", "Status"]))
    # Mark bad channels
    raw.info["bads"].extend(bads)
    sfreq = raw.info["sfreq"]

    # Apply notch filter (50 Hz for power line noise)
    raw.notch_filter(50)

    # Set montage and reference
    raw.set_montage("biosemi128", on_missing="ignore")
    raw.set_eeg_reference(ref_channels=["EXG1", "EXG2"])

    # Plot raw data if requested
    if raw_plot:
        raw.plot(block=True, title="Carefully identify wrong channels", theme="light")
        plt.show()
        print("Bads Pre-Interpolation:", raw.info["bads"])

    # Interpolate bad channels
    if interpolate and len(raw.info["bads"]) > 0:
        print(f"Interpolating {len(raw.info['bads'])} bad channel(s): {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True)
        # print("Interpolation completed!")

    # Apply bandpass filter
    raw.filter(lowpass_cut, highpass_cut, l_trans_bandwidth=1, h_trans_bandwidth=1)

    # Resample
    nfreq = 500
    raw.resample(nfreq)

    # Plot filtered data if requested
    if filtered_plot:
        raw.plot(block=True, title="Filtered Signal", theme="light")
        plt.show()

    # Plot PSD if requested
    if psd_plot:
        fig = raw.compute_psd().plot(average=True)
        plt.show()

    # Add marks if requested
    if edit_marks:
        raw_original = raw.copy()
        raw_marked = raw.copy()

        last_ch = len(raw.ch_names) - 1
        old_separation = 30.0
        new_separation = 5.0
        coef = old_separation / new_separation

        previous_value = None  # previous value in vector raw._data[last_ch]
        last_mark = 0.0  # last mark (value different from zero)
        count_zeros = 0.0  # cant of zeros since last mark
        inserted_POS = None  # position where last inserted mark was placed
        last_mark_POS = 0  # position where last original mark was founded
        current_POS = 0  # position counter
        flag = 0.0
        # plt.plot(raw._data[last_ch])
        # plt.show()
        for x in raw_marked._data[last_ch]:
            if x < 255:
                if x != 0:
                    if x != previous_value:
                        # if x == last_mark:
                        separation = (
                            count_zeros // coef
                        )  # separation between two equals marks will be divided by the coef
                        # samples. 5 seconds * 256 samples/second
                        separation = new_separation * raw_marked.info["sfreq"]
                        inserted_POS = last_mark_POS

                        for i in range(last_mark_POS, current_POS - int(separation / 2)):
                            if (i - inserted_POS) >= separation:
                                raw_marked._data[last_ch][i] = last_mark
                                inserted_POS = i
                        last_mark = x
                        last_mark_POS = current_POS
                    count_zeros = 0
                else:
                    count_zeros = count_zeros + 1
            else:
                raw_marked._data[last_ch][current_POS] = 0.0

            previous_value = x
            current_POS = current_POS + 1
        # plt.plot(raw._data[last_ch])
        # plt.show()
        events_original = mne.find_events(raw_original, stim_channel="Status")
        events_marked = mne.find_events(raw_marked, stim_channel="Status")
        # print("raw_marked", events_marked)
        # print("raw_original", events_original)

    return raw_original, raw_marked


def make_ICA(
    raw,
    method: str,
    n_components: int,
    decim: int,
    random_state: int,
    reject_limit,
    bad_ica_channels,
    plot_ica_topo: bool,
    plot_ica_time: bool,
    plot_raw: bool,
):
    """
    Ajusta ICA sobre canales EEG y aplica la exclusión manual de componentes.
    """

    raw_clean = raw.copy()

    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    if method == "infomax":
        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=random_state,
            max_iter="auto",
        )
    else:
        ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter="auto")

    reject = dict(eeg=reject_limit) if reject_limit is not None else None

    ica.fit(raw_clean, picks=picks_eeg, decim=decim, reject=reject)

    if plot_ica_topo:
        ica.plot_components(title="ICA - Componentes Topográficos", cmap="coolwarm")
        plt.show()

    if plot_ica_time:
        ica.plot_sources(raw_clean, title="ICA - Componentes en el Tiempo")
        plt.show()

    if bad_ica_channels is not None:
        ica.exclude = bad_ica_channels

    raw_clean = ica.apply(raw_clean)

    if plot_raw:
        raw.plot(title="Señal cruda antes de ICA")
        raw_clean.plot(title="Señal limpia después de ICA")
        plt.show()

    return ica, raw_clean


# ============================================================================
# DEFINICION DE VARIABLES
# ============================================================================


s = 26
subjects = [
    "01_test_2023",
    "02_test_2023",
    "04_test_2023",
    "05_test_2023",
    "06_test_2023",
    "07_test_2023",
    "10_test_2023",
    "11_test_2023",
    "12_test_2023",  # s= 8 - 12_test_2023 ANULADO
    "13_test_2023",
    "14_test_2023",
    "15_test_2023",
    "16_test_2023",
    "20_test_2023",
    "21_test_2023",
    "22_test_2023",
    "23_test_2023",
    "24_test_2023",
    "25_test_2023",
    "26_test_2023",
    "28_test_2023",
    "29_test_2023",
    "30_test_2023",
    "30_test_2023_bis",
    "31_test_2023",
    "34_test_2023",
    "30_test_2023_merged_raw",
]

bads_preICA = [
    ["B2", "C4", "C30", "D4", "D5", "D10", "D12"],  # s=0 - 01
    ["C10", "D8", "D9", "D24", "D25"],  # s=1 - 02
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=2 - 04
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=3 - 05
    ["B6", "D10", "D25", "D29"],  # s=4 - 06
    ["C8"],  # s=5 - 07
    ["A10", "A17", "A27", "B4", "B30", "B31", "C17", "C29", "C32", "D19"],  # s=6 - 10
    ["B13"],  # s= 7 - 11_test_2023 RARISIMO EL PSD
    [],  # s= 8 - 12_test_2023 ANULADO
    ["A23", "B20", "B21", "B23", "C16", "C29", "C30", "C32", "D3", "D23"],  # s=9 - 13
    ["A6", "A7", "A12", "A13", "A26", "D22", "D23"],  # s= 10 - 14_test_2023
    ["A9", "A30", "B7", "C21", "C22", "D18", "C8", "C14"],  # s = 11 - 15_test_2023
    ["B24", "D31", "D32"],  # s = 12 - 16_test_2023
    ["A6", "D3", "D27"],  # s = 13 - 20_test_2023
    ["A12", "A13", "A14", "B8", "B9", "B28", "D11", "D23"],  # s = 14 - 21_test_2023
    ["A32", "C4", "C5", "C6", "C7", "C8", "C16", "C17", "C29"],  # s = 15 - 22_test_2023
    ["C14", "C15", "D4"],  # s = 16 - 23_test_2023 #MUY FEO.
    ["A32", "B9", "B27", "C16", "D20", "D30", "D32"],
    ["A10", "A24", "A25", "A32", "B23", "B25", "C18", "C23", "C24", "C28", "D31", "D32"],  # s = 18 - 25_test_2023
    ["A17", "A25", "C6", "D17", "D19", "D22", "D23", "D24", "D28", "D32"],  # s = 19 - 26_test_2023
    ["B8", "B9", "B26", "C16"],  # s = 20 - 28_test_2023
    ["A14", "A21", "A22", "A31", "B3", "B4", "B24", "C2", "C23", "D22", "D23"],  # s = 21 - 29_test_2023
    ["A6", "A15", "B1", "B13", "C26", "D3"],  # s = 22 - 30_test_2023
    ["A15", "A20", "B6", "B13", "C30", "D3"],  # s = 23 - 30_test_2023_bis
    ["A11", "B21", "C16", "C17", "C29", "C30", "D5"],  # s = 24 - 31_test_2023
    ["A24", "B1", "B2", "B18", "B19", "B28", "B32", "C6", "C28", "D8", "D11", "D12", "D26"],  # s = 25 - 34_test_2023
    ["A6", "A8", "A15", "B24", "C16", "D3"],
]

bads_ICA = [
    [0, 4, 13, 14],  # s = 0
    [0, 3, 9, 14],  # s = 1
    [0, 2, 12],  # s = 2
    [0, 3, 5, 8, 13],  # s = 3
    [0, 1, 5, 12],  # s = 4
    [0, 3, 6, 8, 9, 11, 14],  # s = 5
    [0, 3, 4, 10, 11, 12, 13],  # s = 6
    [0, 1, 6, 14],  # s = 7
    [],  # s = 8
    [1, 4, 10, 12],  # s = 9
    [0, 2, 6, 10, 12, 13],  # s = 10
    [0, 5, 9, 11],  # s = 11
    [0, 3, 4, 8, 13],  # s = 12
    [0, 1, 9, 14],  # s = 13
    [0, 2, 5, 12, 13, 14],  # s = 14
    [0, 1, 3, 13, 14],  # s = 15
    [0, 1, 2, 5, 11, 13, 14],  # s = 16
    [0, 3, 4, 11, 12, 13],  # s = 17
    [1, 7, 8, 10, 14],  # s = 18
    [0, 2, 3, 8, 9],  # s = 19
    [0, 1, 3, 4, 10],  # s = 20
    [0, 3, 7, 11, 12, 14],  # s = 21
    [],  # s = 22
    [],  # s = 23
    [0, 4, 6, 8, 11, 14],  # s = 24
    [0, 2, 3, 5, 7, 14],  # s = 25
    [1, 4, 5, 8, 11, 13],
]

print("#s =", s, "-", subjects[s])
print("bads_preICA:", bads_preICA[s])
print("bads_ICA:", bads_ICA[s])


# ============================================================================
# LLAMADO DE FUNCIONES
# ============================================================================

file = subjects[s]
bads = bads_preICA[s]
path = "E:/Doctorado/protocol2023/"  # Gamer
# path = "D:/Doctorado/protocol2023/"  # Lenovo
# path = "/media/pedro/Expansion/Doctorado/protocol2023/" #Ideapad
excluded = [
    "EXG3",
    "EXG4",
    "EXG5",
    "EXG6",
    "EXG7",
    "EXG8",  # External electrodes
]

raw_original, raw_marked = preprocessing_mne(
    path=path,
    file=file,
    excluded=excluded,
    bads=bads,
    lowpass_cut=1,  # High-pass filter: remove slow drifts < 1 Hz
    highpass_cut=30,  # Low-pass filter: remove noise > 30 Hz
    raw_plot=False,  # Set True to visually inspect raw data
    filtered_plot=False,  # Set True to see filtered data
    psd_plot=False,  # Set True to see power spectrum
    edit_marks=True,  # Set True to add time markers
    interpolate=True,  # Interpolate bad channels
)

ica, raw_clean = make_ICA(
    raw_original,
    method="infomax",
    n_components=15,
    decim=3,
    random_state=23,
    reject_limit=250e-6,
    bad_ica_channels=bads_ICA[s],
    plot_ica_topo=True,
    plot_ica_time=True,
    plot_raw=True,
)
