"""
AgriNusa - Sistem Prediksi Kelayakan Lahan Padi
Fitur dan tipe input dideteksi otomatis dari preprocessor pipeline.

Prasyarat file di folder yang sama:
    - rf_pipeline_paddy.joblib
    - lr_pipeline_paddy.joblib
    - fitur_aplikasi.joblib

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(
    page_title="AgriNusa - Prediksi Kelayakan Lahan",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f4f6f4 !important;
    color: #1a1a1a !important;
}

.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1140px !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Selectbox — paksa light mode ── */
[data-baseweb="select"] {
    color-scheme: light !important;
}

[data-baseweb="select"] > div:first-child {
    background-color: #ffffff !important;
    border: 1.5px solid #bdd4bf !important;
    border-radius: 7px !important;
    color: #1a1a1a !important;
}

[data-baseweb="select"] > div:first-child:focus-within {
    border-color: #2d6a35 !important;
    box-shadow: 0 0 0 3px rgba(45,106,53,0.12) !important;
}

/* Teks nilai terpilih */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] div[class*="placeholder"],
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] span,
[data-baseweb="select"] p {
    color: #1a1a1a !important;
    font-size: 0.86rem !important;
}

/* Dropdown popup */
[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 1px solid #bdd4bf !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}

[data-baseweb="menu"] {
    background-color: #ffffff !important;
}

[data-baseweb="menu"] ul {
    background-color: #ffffff !important;
}

[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    font-size: 0.86rem !important;
}

[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #edf7ee !important;
    color: #1c4a24 !important;
}

/* ── Number input ── */
[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #bdd4bf !important;
    border-radius: 7px !important;
    overflow: hidden;
}

[data-baseweb="input"] > div:focus-within {
    border-color: #2d6a35 !important;
    box-shadow: 0 0 0 3px rgba(45,106,53,0.12) !important;
}

[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.86rem !important;
}

/* Tombol stepper +/- */
[data-baseweb="input"] button {
    background-color: #f0f5f0 !important;
    border-left: 1px solid #d0e0d2 !important;
    color: #2d6a35 !important;
}

/* ── Label widget ── */
label[data-testid="stWidgetLabel"] p {
    color: #2e3d30 !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
}

/* ── Radio button ── */
[data-testid="stRadio"] label p {
    color: #1a1a1a !important;
    font-size: 0.86rem !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    background: #1c4a24 !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    letter-spacing: 0.03em;
    transition: background 0.15s ease !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #2d6a35 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #dde8de !important;
    margin: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)



def card_open(judul: str = ""):
    """Buka div kartu putih dengan judul opsional."""
    judul_html = (
        f'<p style="font-size:0.68rem;font-weight:700;color:#2d6a35;'
        f'letter-spacing:0.12em;text-transform:uppercase;'
        f'border-bottom:1px solid #dde8de;padding-bottom:0.6rem;'
        f'margin-bottom:1rem;">{judul}</p>'
        if judul else ""
    )
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #dde8de;'
        f'border-radius:12px;padding:1.4rem 1.6rem 1.2rem;">'
        f'{judul_html}',
        unsafe_allow_html=True,
    )


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def label_kecil(teks: str):
    st.markdown(
        f'<p style="font-size:0.68rem;font-weight:700;color:#2d6a35;'
        f'letter-spacing:0.12em;text-transform:uppercase;'
        f'border-bottom:1px solid #dde8de;padding-bottom:0.6rem;'
        f'margin:0 0 1rem 0;">{teks}</p>',
        unsafe_allow_html=True,
    )



@st.cache_resource
def load_assets():
    hasil = {"rf": None, "lr": None, "fitur": [], "error": []}
    for key, fname in [("rf",    "rf_pipeline_paddy.joblib"),
                       ("lr",    "lr_pipeline_paddy.joblib"),
                       ("fitur", "fitur_aplikasi.joblib")]:
        path = Path(fname)
        if path.exists():
            try:
                hasil[key] = joblib.load(path)
            except Exception as exc:
                hasil["error"].append(f"{fname}: {exc}")
        else:
            hasil["error"].append(f"{fname} tidak ditemukan.")
    return hasil


assets     = load_assets()
model_rf   = assets["rf"]
model_lr   = assets["lr"]
fitur_list = assets["fitur"] if isinstance(assets["fitur"], list) else []
DEMO_MODE  = bool(assets["error"])



def ekstrak_info_fitur(pipeline, fitur: list) -> dict:
    """
    Baca tipe dan opsi tiap fitur dari ColumnTransformer di dalam pipeline.
    Return: {nama_fitur: {"tipe": "kategorik"|"numerik", "opsi": list|None}}
    """
    info = {}
    try:
        prep = pipeline.named_steps["prep"]
        for nama_tf, transformer, kolom in prep.transformers_:
            for col in kolom:
                if col not in fitur:
                    continue
                if nama_tf == "cat":
                    idx  = list(kolom).index(col)
                    info[col] = {
                        "tipe": "kategorik",
                        "opsi": list(transformer.categories_[idx]),
                    }
                elif nama_tf == "num":
                    info[col] = {"tipe": "numerik", "opsi": None}
    except Exception:
        pass

    for col in fitur:
        if col not in info:
            info[col] = {"tipe": "numerik", "opsi": None}

    return info


pipeline_ref = model_rf if model_rf is not None else model_lr
info_fitur: dict = {}
if pipeline_ref is not None and fitur_list:
    info_fitur = ekstrak_info_fitur(pipeline_ref, fitur_list)


st.markdown("""
<div style="background:#ffffff;border:1px solid #dde8de;border-radius:12px;
            padding:1.2rem 1.6rem;margin-bottom:1.5rem;
            display:flex;align-items:center;justify-content:space-between;">
    <div>
        <p style="font-family:'DM Serif Display',serif;font-size:1.75rem;
                  color:#1c4a24;margin:0;line-height:1.1;">AgriNusa</p>
        <p style="font-size:0.82rem;color:#6b7c6d;margin:0.15rem 0 0;">
            Sistem Prediksi Kelayakan Lahan Padi</p>
    </div>
    <span style="font-size:0.7rem;font-weight:700;color:#2d6a35;
                 background:#edf7ee;border:1px solid #b8d9bc;border-radius:5px;
                 padding:5px 13px;letter-spacing:0.08em;text-transform:uppercase;">
        Alat Bantu Penyuluh Pertanian
    </span>
</div>
""", unsafe_allow_html=True)

if DEMO_MODE:
    st.warning(
        "Beberapa file model tidak ditemukan: " + " | ".join(assets["error"]) +
        "  \nLetakkan file `.joblib` di folder yang sama dengan `app.py`.",
        icon=None,
    )



col_form, col_result = st.columns([6, 4], gap="large")


with col_form:
    with st.form(key="form_prediksi"):

        # -- Pemilihan model --
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde8de;border-radius:12px;
                    padding:1.2rem 1.5rem 1rem;">
            <p style="font-size:0.68rem;font-weight:700;color:#2d6a35;
                      letter-spacing:0.12em;text-transform:uppercase;
                      border-bottom:1px solid #dde8de;padding-bottom:0.6rem;
                      margin:0 0 0.9rem 0;">Pilih Model Prediksi</p>
        """, unsafe_allow_html=True)

        pilihan_model = st.radio(
            label="model",
            options=["Random Forest", "Logistic Regression"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # -- Input fitur --
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde8de;border-radius:12px;
                    padding:1.2rem 1.5rem 0.5rem;">
            <p style="font-size:0.68rem;font-weight:700;color:#2d6a35;
                      letter-spacing:0.12em;text-transform:uppercase;
                      border-bottom:1px solid #dde8de;padding-bottom:0.6rem;
                      margin:0 0 1rem 0;">Data Lahan dan Rencana Tanam</p>
        """, unsafe_allow_html=True)

        nilai_input = {}

        if not fitur_list:
            st.info("Daftar fitur tidak tersedia. Pastikan `fitur_aplikasi.joblib` ada.")
        else:
            tengah     = int(np.ceil(len(fitur_list) / 2))
            grup_kiri  = fitur_list[:tengah]
            grup_kanan = fitur_list[tengah:]

            kol_ki, kol_ka = st.columns(2, gap="medium")

            def render_input(kolom_st, nama_fitur: str):
                """Render widget sesuai tipe: selectbox atau number_input."""
                meta = info_fitur.get(nama_fitur, {"tipe": "numerik", "opsi": None})
                with kolom_st:
                    if meta["tipe"] == "kategorik" and meta["opsi"]:
                        return st.selectbox(
                            label=nama_fitur,
                            options=meta["opsi"],
                            key=f"inp_{nama_fitur}",
                        )
                    return st.number_input(
                        label=nama_fitur,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"inp_{nama_fitur}",
                    )

            for f in grup_kiri:
                nilai_input[f] = render_input(kol_ki, f)
            for f in grup_kanan:
                nilai_input[f] = render_input(kol_ka, f)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Jalankan Prediksi",
            use_container_width=True,
        )


with col_result:

    # Kontainer hasil selalu tampil sebagai kartu putih
    with st.container():
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde8de;border-radius:12px;
                    padding:1.2rem 1.5rem 1.2rem;">
            <p style="font-size:0.68rem;font-weight:700;color:#2d6a35;
                      letter-spacing:0.12em;text-transform:uppercase;
                      border-bottom:1px solid #dde8de;padding-bottom:0.6rem;
                      margin:0 0 1rem 0;">Hasil Prediksi</p>
        """, unsafe_allow_html=True)

        if not submitted:
            st.markdown("""
            <div style="border:1.5px dashed #c3d9c6;border-radius:9px;
                        padding:2.2rem 1rem;text-align:center;color:#8aaa8c;">
                <p style="font-size:0.84rem;margin:0;line-height:1.8;">
                    Isi data lahan di sebelah kiri,<br>
                    lalu tekan <strong>Jalankan Prediksi</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            model_aktif = model_rf if pilihan_model == "Random Forest" else model_lr

            if model_aktif is None or not nilai_input:
                st.warning("Model tidak tersedia. Periksa kembali file joblib.")

            else:
                df_input = pd.DataFrame([nilai_input])[fitur_list]

                try:
                    prediksi     = model_aktif.predict(df_input)[0]
                    probabilitas = model_aktif.predict_proba(df_input)[0]
                    kelas_model  = list(model_aktif.classes_)

                    CLASS_ORDER = ["Rendah", "Sedang", "Tinggi"]
                    prob_dict   = dict(zip(kelas_model, probabilitas))
                    prob_urut   = [prob_dict.get(k, 0.0) for k in CLASS_ORDER]

                    STYLE = {
                        "Tinggi": {
                            "bg": "#f0faf1", "border": "#4caf50",
                            "judul_warna": "#2d6a35", "teks_warna": "#1c4a24",
                            "narasi_warna": "#2d5c32", "bar": "#2d6a35",
                            "status": "Layak - Produktivitas Tinggi",
                        },
                        "Sedang": {
                            "bg": "#fffcf0", "border": "#dba416",
                            "judul_warna": "#8a6200", "teks_warna": "#6b4800",
                            "narasi_warna": "#6b4800", "bar": "#c48a00",
                            "status": "Perlu Optimasi",
                        },
                        "Rendah": {
                            "bg": "#fff5f5", "border": "#d95252",
                            "judul_warna": "#a02828", "teks_warna": "#7a1f1f",
                            "narasi_warna": "#7a2828", "bar": "#c04040",
                            "status": "Tidak Layak - Produktivitas Rendah",
                        },
                    }

                    DESKRIPSI = {
                        "Tinggi": (
                            "Kondisi lahan dan rencana tanam mendukung "
                            "produktivitas tinggi. Pertahankan jadwal irigasi "
                            "dan lakukan pemantauan hama secara berkala."
                        ),
                        "Sedang": (
                            "Produktivitas berpotensi di bawah optimal. "
                            "Pertimbangkan penyesuaian dosis pupuk susulan "
                            "dan konsultasikan varietas alternatif dengan "
                            "PPL setempat."
                        ),
                        "Rendah": (
                            "Kombinasi kondisi lahan ini diprediksi "
                            "menghasilkan produktivitas rendah. Disarankan "
                            "konsultasi langsung dengan Penyuluh Pertanian "
                            "Lapangan sebelum melanjutkan."
                        ),
                    }

                    s = STYLE[prediksi]

                    # Kotak verdict
                    st.markdown(f"""
                    <div style="background:{s['bg']};border:1.5px solid {s['border']};
                                border-radius:10px;padding:1.3rem 1.4rem 1.1rem;
                                margin-bottom:1.1rem;">
                        <p style="font-size:0.67rem;font-weight:700;
                                  color:{s['judul_warna']};letter-spacing:0.1em;
                                  text-transform:uppercase;margin:0 0 0.25rem;">
                            {s['status']}
                        </p>
                        <p style="font-family:'DM Serif Display',serif;
                                  font-size:1.9rem;color:{s['teks_warna']};
                                  margin:0 0 0.5rem;line-height:1.1;">
                            {prediksi.upper()}
                        </p>
                        <p style="font-size:0.83rem;color:{s['narasi_warna']};
                                  line-height:1.65;margin:0;">
                            {DESKRIPSI[prediksi]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Label probabilitas
                    st.markdown("""
                    <p style="font-size:0.68rem;font-weight:700;color:#2d6a35;
                               letter-spacing:0.12em;text-transform:uppercase;
                               border-top:1px solid #dde8de;padding-top:0.9rem;
                               margin:0 0 0.8rem;">Distribusi Probabilitas</p>
                    """, unsafe_allow_html=True)

                    # Bar probabilitas
                    for kelas, prob in zip(CLASS_ORDER, prob_urut):
                        pct    = prob * 100
                        bold   = "700" if kelas == prediksi else "400"
                        warna  = STYLE[kelas]["bar"]
                        st.markdown(f"""
                        <div style="margin-bottom:0.65rem;">
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.82rem;font-weight:{bold};
                                        color:#2e3d30;margin-bottom:4px;">
                                <span>{kelas}</span>
                                <span>{pct:.1f}%</span>
                            </div>
                            <div style="height:8px;background:#e4ede5;
                                        border-radius:99px;overflow:hidden;">
                                <div style="width:{pct:.1f}%;height:100%;
                                            background:{warna};border-radius:99px;">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Info model
                    st.markdown(
                        f'<p style="font-size:0.71rem;color:#9aaa9c;margin-top:0.7rem;">'
                        f'Model: {pilihan_model} &nbsp;|&nbsp; '
                        f'Keyakinan: {max(prob_urut)*100:.1f}%</p>',
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.error(f"Prediksi gagal. Detail: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)



st.markdown("""
<div style="border-top:1px solid #dde8de;margin-top:2rem;padding-top:0.8rem;
            font-size:0.72rem;color:#9aaa9c;text-align:center;">
    AgriNusa &mdash; Sistem Prediksi Kelayakan Lahan Padi &nbsp;|&nbsp;
    Alat Bantu Penyuluh Pertanian
</div>
""", unsafe_allow_html=True)
