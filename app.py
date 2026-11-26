"""Tampilan Streamlit: Simulasi TSP (Nearest Neighbor, Dynamic Programming & Manual)."""
import random
import re
import matplotlib.pyplot as plt
import streamlit as st
from tsp_nn import (
    bangun_langkah_manual,
    buat_data,
    buat_koordinat_tersebar,
    nearest_neighbor,
    tsp_dp,
)
from visualisasi import gambar_graf

MAX_NODE = 14

st.set_page_config(page_title="Simulasi TSP - NN vs Dynamic Programming", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    [data-testid="stHeader"] { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #DBEAFE;
        width: 225px !important;
        min-width: 225px !important;
        max-width: 225px !important;
        flex: 0 0 225px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 225px !important;
        min-width: 225px !important;
    }

    /* Teks umum & tabel */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stTable"],
    [data-testid="stRadio"] * {
        color: #1E293B !important;
    }
    .stApp [data-testid="stTable"] table,
    .stApp [data-testid="stTable"] th,
    .stApp [data-testid="stTable"] td {
        color: #1E293B !important;
        background-color: #EFF6FF !important;
        border-color: #BFDBFE !important;
    }
    .stApp [data-testid="stTable"] th {
        background-color: #DBEAFE !important;
        font-weight: 600;
    }

    /* Kotak Random Seed (stNumberInput) */
    [data-testid="stNumberInputContainer"],
    [data-testid="stNumberInputContainer"] > div,
    [data-testid="stNumberInputContainer"] input {
        background-color: #EFF6FF !important;
        border-color: #93C5FD !important;
        color: #1E3A8A !important;
        -webkit-text-fill-color: #1E3A8A !important;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
        background-color: #DBEAFE !important;
        color: #1E3A8A !important;
        border-left: 1px solid #93C5FD !important;
    }
    [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInputStepUp"] svg {
        fill: #1E3A8A !important;
    }

    :root, .stApp, [data-testid="stSidebar"] {
        --background-color: #FFFFFF !important;
        --secondary-background-color: #EFF6FF !important;
        --text-color: #1E3A8A !important;
        --primary-color: #1F77B4 !important;
    }

    /* Kotak & Menu Dropdown (stSelectbox) */
    html body .stApp [data-testid="stSelectbox"] [data-baseweb="select"],
    html body .stApp [data-testid="stSelectbox"] [data-baseweb="select"] div,
    html body .stApp [data-testid="stSelectbox"] [data-baseweb="select"] input,
    html body .stApp [data-baseweb="select"],
    html body .stApp [data-baseweb="select"] div,
    html body .stApp [data-baseweb="select"] input {
        background: #EFF6FF !important;
        background-color: #EFF6FF !important;
        border-color: #93C5FD !important;
        color: #1E3A8A !important;
        -webkit-text-fill-color: #1E3A8A !important;
    }
    html body .stApp [data-baseweb="select"] > div:focus-within {
        border-color: #1F77B4 !important;
        box-shadow: 0 0 0 1px #1F77B4 !important;
    }
    html body .stApp [data-baseweb="select"] span,
    html body .stApp [data-baseweb="select"] p {
        color: #1E3A8A !important;
        -webkit-text-fill-color: #1E3A8A !important;
    }
    html body .stApp [data-baseweb="select"] svg {
        fill: #1E3A8A !important;
        color: #1E3A8A !important;
    }
    html body [data-baseweb="popover"],
    html body [data-baseweb="popover"] > div,
    html body [data-baseweb="popover"] ul,
    html body [data-baseweb="menu"],
    html body [role="listbox"] {
        background-color: #EFF6FF !important;
        border: 1px solid #93C5FD !important;
    }
    html body [data-baseweb="menu"] li,
    html body [data-baseweb="menu"] li *,
    html body [role="option"],
    html body [role="option"] * {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
    }
    html body [data-baseweb="menu"] li:hover,
    html body [data-baseweb="menu"] li:hover *,
    html body [data-baseweb="menu"] li[aria-selected="true"],
    html body [data-baseweb="menu"] li[aria-selected="true"] *,
    html body [role="option"]:hover,
    html body [role="option"]:hover *,
    html body [role="option"][aria-selected="true"],
    html body [role="option"][aria-selected="true"] * {
        background-color: #DBEAFE !important;
        color: #1E3A8A !important;
    }

    /* Tombol Utama */
    .stApp [data-testid="stBaseButton-primary"],
    .st-key-tsp_controls [data-testid="stBaseButton-primary"] {
        background-color: #1F77B4 !important;
        border: 1px solid #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(31, 119, 180, 0.28) !important;
    }
    .stApp [data-testid="stBaseButton-primary"] *,
    .st-key-tsp_controls [data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
    }
    .stApp [data-testid="stBaseButton-primary"]:hover,
    .st-key-tsp_controls [data-testid="stBaseButton-primary"]:hover {
        background-color: #186296 !important;
        border-color: #1E40AF !important;
        color: #FFFFFF !important;
    }

    /* Tombol Sekunder */
    .stApp [data-testid="stBaseButton-secondary"],
    .st-key-tsp_controls [data-testid="stBaseButton-secondary"] {
        background-color: #EFF6FF !important;
        border: 1px solid #93C5FD !important;
        color: #1E3A8A !important;
        font-weight: 500;
    }
    .stApp [data-testid="stBaseButton-secondary"]:hover,
    .st-key-tsp_controls [data-testid="stBaseButton-secondary"]:hover {
        background-color: #DBEAFE !important;
        border-color: #60A5FA !important;
        color: #1E3A8A !important;
    }
    .stApp button:disabled {
        background-color: #F0F9FF !important;
        border-color: #E0F2FE !important;
        color: #94A3B8 !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }
    .stApp button:disabled * {
        color: #94A3B8 !important;
    }

    /* Toolbar & tombol fullscreen di pojok gambar graf */
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbar"] div,
    [data-testid="stElementToolbar"] button,
    [data-testid="stElementToolbarButton"],
    [data-testid="StyledFullScreenButton"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #1E3A8A !important;
    }
    [data-testid="stElementToolbar"] svg,
    [data-testid="StyledFullScreenButton"] svg {
        fill: #1E3A8A !important;
        color: #1E3A8A !important;
        transition: transform 0.15s ease, opacity 0.15s ease;
    }
    [data-testid="stElementToolbar"] button:hover svg,
    [data-testid="StyledFullScreenButton"]:hover svg {
        transform: scale(1.15);
        opacity: 0.8;
    }

    /* Kotak status custom & semua alert bawaan */
    .tsp-status,
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    [data-baseweb="notification"] {
        background-color: #EFF6FF !important;
        border: 1px solid #93C5FD !important;
        border-radius: 0.5rem !important;
        color: #1E3A8A !important;
        line-height: 1.5;
    }
    .tsp-status { padding: 0.75rem 1rem; }
    .tsp-status *,
    [data-testid="stAlert"] *,
    [data-baseweb="notification"] * {
        color: #1E3A8A !important;
    }

    /* Warna aksen slider Jumlah lokasi agar ikut biru */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #1F77B4 !important;
    }
    [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMin"] {
        color: #1E3A8A !important;
    }

    /* Kartu ber-border untuk section Graf Lokasi */
    .st-key-graf_card,
    .st-key-graf_card [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.5px solid #BFDBFE !important;
        border-radius: 0.75rem !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.04) !important;
    }

    .tsp-legend { color: #1E293B; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_teks(teks):
    """Pastikan semua angka desimal di dalam string berformat 1 angka di belakang koma."""
    return re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.1f}", str(teks))


# ---------- state & callback ----------
def reset():
    st.session_state.mulai = False
    st.session_state.langkah = 0
    st.session_state.rute_manual = []
    st.session_state.auto = False
    st.session_state.hasil_tersimpan = False


def mulai():
    st.session_state.mulai = True
    st.session_state.langkah = 0
    st.session_state.rute_manual = []
    st.session_state.hasil_tersimpan = False


def berikutnya():
    st.session_state.langkah += 1


def sebelumnya():
    if st.session_state.get("mode_simulasi") == "Pilih Node Sendiri":
        if len(st.session_state.rute_manual) > 1:
            st.session_state.rute_manual.pop()
            st.session_state.hasil_tersimpan = False
    else:
        if st.session_state.langkah > 0:
            st.session_state.langkah -= 1
            st.session_state.hasil_tersimpan = False


def pilih_node_manual(node_idx):
    st.session_state.rute_manual.append(node_idx)


def otomatis():
    st.session_state.mulai = True
    st.session_state.auto = True


def seed_berubah():
    st.session_state.riwayat_rute = []
    reset()


def acak_posisi():
    seed_lama = st.session_state.random_seed
    seed_baru = random.SystemRandom().randint(0, 2**31 - 1)
    if seed_baru == seed_lama:
        seed_baru = (seed_baru + 1) % (2**31)
    st.session_state.random_seed = seed_baru
    seed_berubah()


if "mulai" not in st.session_state:
    reset()
if "rute_manual" not in st.session_state:
    st.session_state.rute_manual = []
if "riwayat_rute" not in st.session_state:
    st.session_state.riwayat_rute = []
if "hasil_tersimpan" not in st.session_state:
    st.session_state.hasil_tersimpan = False
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

# ---------- data & algoritma ----------
st.title("Simulasi Rute Pengangkutan Sampah (TSP)")
mode = st.sidebar.radio(
    "Mode Simulasi",
    ["Nearest Neighbor", "Dynamic Programming", "Pilih Node Sendiri"],
    key="mode_simulasi",
    on_change=reset,
)
mode_manual = mode == "Pilih Node Sendiri"

jumlah = st.sidebar.slider("Jumlah lokasi", 4, MAX_NODE, 8)
if st.session_state.get("jumlah") != jumlah:
    st.session_state.jumlah = jumlah
    reset()
    st.session_state.riwayat_rute = []

st.session_state.jumlah = jumlah
seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=2**31 - 1,
    step=1,
    key="random_seed",
    on_change=seed_berubah,
)
label = [f"LOC{i + 1}" for i in range(jumlah)]
depot = st.sidebar.selectbox(
    "Titik Awal (Depot)",
    label,
    key="depot",
    on_change=reset,
)
st.sidebar.button(
    "Acak Posisi Lokasi",
    on_click=acak_posisi,
    use_container_width=True,
)
st.sidebar.caption(
    "Data simulasi menggunakan koordinat buatan tersebar dan jarak Euclidean."
)

koord_seed = buat_koordinat_tersebar(MAX_NODE, seed)
label, koord, jarak = buat_data(jumlah, koord_seed)
awal = label.index(depot)

if st.session_state.mulai and not st.session_state.rute_manual:
    st.session_state.rute_manual = [awal]

if mode == "Dynamic Programming":
    langkah_algo, waktu_ms = tsp_dp(jarak, label, awal=awal)
else:
    langkah_algo, waktu_ms = nearest_neighbor(jarak, label, awal=awal)

total_langkah = len(langkah_algo)
teks_runtime = "-" if mode_manual else f"{waktu_ms:.4f} ms"

if mode_manual:
    langkah_aktif = bangun_langkah_manual(
        st.session_state.rute_manual, jarak, label, awal
    )
    m = len(langkah_aktif)
else:
    langkah_aktif = langkah_algo
    m = st.session_state.langkah

selesai = st.session_state.mulai and m >= total_langkah
if selesai and not st.session_state.hasil_tersimpan:
    akhir = langkah_aktif[-1]
    nama_mode = "Manual" if mode_manual else mode
    entri_baru = {
        "Mode": nama_mode,
        "Node": jumlah,
        "Seed": seed,
        "Titik Awal": depot,
        "Rute Terbentuk": " → ".join(label[i] for i in akhir["rute"]),
        "Total Jarak (km)": f"{akhir['total']:.1f}",
        "Runtime": teks_runtime,
    }
    indeks_sama = next(
        (
            idx
            for idx, item in enumerate(st.session_state.riwayat_rute)
            if item["Mode"] == nama_mode
            and item["Node"] == jumlah
            and item["Seed"] == seed
            and item["Titik Awal"] == depot
            and item["Rute Terbentuk"] == entri_baru["Rute Terbentuk"]
        ),
        None,
    )
    if indeks_sama is None:
        st.session_state.riwayat_rute.append(entri_baru)
    else:
        st.session_state.riwayat_rute[indeks_sama] = entri_baru
    st.session_state.hasil_tersimpan = True

# ---------- kontrol ----------
with st.container(key="tsp_controls"):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.button(
        "Mulai Simulasi",
        on_click=mulai,
        disabled=st.session_state.mulai,
        type="primary" if not st.session_state.mulai else "secondary",
        use_container_width=True,
    )
    k2.button(
        "Langkah Sebelumnya",
        on_click=sebelumnya,
        disabled=(not st.session_state.mulai) or m == 0 or st.session_state.auto,
        use_container_width=True,
    )
    k3.button(
        "Langkah Berikutnya",
        on_click=berikutnya,
        disabled=(
            (not st.session_state.mulai)
            or selesai
            or st.session_state.auto
            or mode_manual
        ),
        type=(
            "primary"
            if st.session_state.mulai
            and not selesai
            and not st.session_state.auto
            and not mode_manual
            else "secondary"
        ),
        use_container_width=True,
    )
    k4.button(
        "Jalankan Otomatis",
        on_click=otomatis,
        disabled=selesai or st.session_state.auto or mode_manual,
        use_container_width=True,
    )
    k5.button("Reset", on_click=reset, use_container_width=True)

# ---------- status saat ini ----------
if not st.session_state.mulai:
    rute, langkah_kini = [], None
else:
    langkah_kini = langkah_aktif[m - 1] if m > 0 else None
    rute = langkah_kini["rute"] if langkah_kini else [awal]

kandidat_preview = None
belum_preview = []
if st.session_state.mulai and m < total_langkah:
    if mode_manual:
        curr_idx = rute[-1]
        sisa_idx = [j for j in range(jumlah) if j not in rute]
        if not sisa_idx:
            sisa_idx = [awal]
        belum_preview = sisa_idx
        kandidat_preview = {j: jarak[curr_idx][j] for j in sisa_idx}
    else:
        belum_preview = langkah_algo[m]["belum"]
        kandidat_preview = langkah_algo[m]["kandidat"]

total_jarak = langkah_kini["total"] if langkah_kini else 0.0
rute_metrik = langkah_kini["rute"] if langkah_kini else [awal]
lokasi_dikunjungi = len(set(rute_metrik)) if st.session_state.mulai else 0
posisi = label[rute_metrik[-1]]

# Perbesar proporsi kolom kiri (kanvas graf) dan perkecil kolom kanan (panel langkah)
kol_graf, kol_info = st.columns([1.55, 1.0], gap="medium")

with kol_graf:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Jarak", f"{total_jarak:.1f} km")
    m2.metric("Lokasi Dikunjungi", f"{lokasi_dikunjungi} / {jumlah}")
    m3.metric("Posisi Saat Ini", posisi)
    m4.metric("Runtime", teks_runtime)

    with st.container(key="graf_card", border=True):
        st.subheader("Graf Lokasi")
        fig = gambar_graf(
            label, koord, jarak, rute if rute else [awal], kandidat_preview
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            """
            <style>
            .tsp-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 18px;
                font-size: 14px;
                border-top: 1px solid #DBEAFE;
                margin-top: 0.5rem;
                padding-top: 0.85rem;
            }
            .tsp-legend-item { display: flex; align-items: center; gap: 7px; }
            .tsp-legend-dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
            .tsp-legend-line { width: 22px; display: inline-block; border-top: 3px solid; }
            .tsp-legend-dashed { border-top-style: dashed; }
            </style>
            <div class="tsp-legend">
              <div class="tsp-legend-item"><span class="tsp-legend-dot" style="background:#FFFFFF;border:2px solid #0F172A"></span>Belum dikunjungi</div>
              <div class="tsp-legend-item"><span class="tsp-legend-dot" style="background:#FEF3C7;border:2px solid #2563EB;box-shadow:0 0 0 2px #93C5FD"></span>Lokasi saat ini</div>
              <div class="tsp-legend-item"><span class="tsp-legend-dot" style="background:#DCFCE7;border:2px solid #16A34A"></span>Sudah dikunjungi</div>
              <div class="tsp-legend-item"><span class="tsp-legend-dot" style="background:#FFFFFF;border:2.5px solid #1D4ED8"></span>Kandidat node</div>
              <div class="tsp-legend-item"><span class="tsp-legend-line" style="border-color:#F97316"></span>Rute terbentuk</div>
              <div class="tsp-legend-item"><span class="tsp-legend-line tsp-legend-dashed" style="border-color:#2563EB"></span>Kandidat sisi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Riwayat Perbandingan Rute & Runtime**")
    if st.session_state.riwayat_rute:
        st.table(st.session_state.riwayat_rute)
    else:
        st.caption("Riwayat hasil dan runtime akan muncul setelah simulasi selesai.")

with kol_info:
    st.subheader("Langkah Simulasi")
    if not st.session_state.mulai:
        st.markdown(
            f'<div class="tsp-status">Tekan <strong>Mulai Simulasi</strong> '
            f'untuk memulai dari depot <strong>{depot}</strong>.</div>',
            unsafe_allow_html=True,
        )
    elif m == 0:
        st.write(f"Lokasi saat ini: **{depot} (depot)**")
        st.write(
            "Belum dikunjungi: "
            + ", ".join(label[j] for j in belum_preview)
        )
        st.write("Total jarak sementara: **0.0 km**")
        if mode_manual:
            st.markdown(
                '<div class="tsp-status">Pilih lokasi tujuan berikutnya di bawah ini.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tsp-status">Tekan <strong>Langkah Berikutnya</strong>.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("**Preview Kandidat Langkah Pertama**")
        st.table([
            {"Tujuan": label[j], "Jarak (km)": f"{d:.1f}"}
            for j, d in kandidat_preview.items()
        ])
    else:
        s = langkah_kini
        st.write(f"**Langkah {m} dari {total_langkah}**")
        st.write(f"Lokasi saat ini: **{label[rute[-1]]}**")
        st.write(
            "Belum dikunjungi: "
            + (", ".join(label[j] for j in belum_preview) or "-")
        )
        st.markdown("**Perbandingan Jarak Langkah Terakhir**")
        st.table([
            {
                "Tujuan": label[j],
                "Jarak (km)": f"{d:.1f}",
                "Status": "Terpilih" if j == s["tujuan"] else "",
            }
            for j, d in s["kandidat"].items()
        ])
        st.success(f"Dipilih: **{label[s['tujuan']]}** ({s['jarak']:.1f} km)")
        st.write(fmt_teks(s["alasan"]))
        st.write(f"Total jarak sementara: **{s['total']:.1f} km**")
        if m < total_langkah and kandidat_preview:
            st.markdown("**Preview Kandidat Berikutnya**")
            st.table([
                {"Tujuan": label[j], "Jarak (km)": f"{d:.1f}"}
                for j, d in kandidat_preview.items()
            ])

    # Kontrol pemilihan node saat mode manual aktif
    if st.session_state.mulai and mode_manual and not selesai and kandidat_preview:
        st.markdown("**Pilih Tujuan Berikutnya**")
        opsi_kandidat = list(kandidat_preview.keys())
        if len(opsi_kandidat) == 1 and opsi_kandidat[0] == awal:
            st.button(
                f"Kembali ke Depot ({label[awal]} - {kandidat_preview[awal]:.1f} km)",
                on_click=pilih_node_manual,
                args=(awal,),
                type="primary",
                use_container_width=True,
            )
        else:
            kol_pil, kol_btn = st.columns([1.3, 1])
            tujuan_pilihan = kol_pil.selectbox(
                "Node tujuan",
                opsi_kandidat,
                format_func=lambda j: f"{label[j]} ({kandidat_preview[j]:.1f} km)",
                label_visibility="collapsed",
                key=f"pilih_node_{m}",
            )
            kol_btn.button(
                "Kunjungi Node",
                on_click=pilih_node_manual,
                args=(tujuan_pilihan,),
                type="primary",
                use_container_width=True,
            )

# ---------- hasil akhir ----------
if selesai:
    st.divider()
    st.subheader("Hasil Akhir")
    akhir = langkah_aktif[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total jarak", f"{akhir['total']:.1f} km")
    c2.metric("Lokasi dikunjungi", jumlah)
    c3.metric("Jumlah ruas", total_langkah)
    c4.metric("Runtime", teks_runtime)
    st.write("**Rute:** " + " → ".join(label[i] for i in akhir["rute"]))
    st.table([
        {
            "Dari": label[s["dari"]],
            "Ke": label[s["tujuan"]],
            "Jarak (km)": f"{s['jarak']:.1f}",
        }
        for s in langkah_aktif
    ])
    if mode_manual:
        st.info(
            "Rute ini disusun berdasarkan pilihan node manual. "
            "Bandingkan total jaraknya dengan Nearest Neighbor dan Dynamic Programming pada tabel Riwayat Perbandingan Rute & Runtime."
        )
    elif mode == "Dynamic Programming":
        st.info(
            "Algoritma Dynamic Programming mengevaluasi seluruh submasalah menggunakan bitmask dan memoisasi "
            "sehingga menjamin rute terpendek secara global dengan kompleksitas O(n²·2ⁿ)."
        )
    else:
        st.info(
            "Pada setiap langkah, algoritma hanya memilih lokasi terdekat saat itu (pendekatan *greedy*, kompleksitas O(n²)). "
            "Karena tidak memikirkan langkah ke depan, hasilnya belum tentu rute terpendek secara global."
        )

# ---------- mode otomatis ----------
if st.session_state.auto and not mode_manual:
    if st.session_state.langkah < total_langkah:
        st.session_state.langkah += 1
        st.rerun()
    else:
        st.session_state.auto = False
        st.rerun()
