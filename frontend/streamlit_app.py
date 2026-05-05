import io
import json
import os
import subprocess
import tempfile
import zipfile

import matplotlib.pyplot as plt
import requests
import streamlit as st

from src.config import get_settings

settings = get_settings()

INTERNAL_API_URL = f"http://{settings.fastapi_host}:{settings.fastapi_port}"
REQUEST_TIMEOUT = 600  # seconds

st.set_page_config(
    page_title="Cavitation Bubble Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Global ── */
    .block-container { padding-top: 2rem; }

    /* ── Login card ── */
    .login-card {
        max-width: 420px;
        margin: 6rem auto 0;
        padding: 2.5rem 2rem 2rem;
        border-radius: 12px;
        background: var(--secondary-background-color);
        box-shadow: 0 4px 24px rgba(0,0,0,.08);
    }
    .login-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .login-header h2 { margin: .4rem 0 0; }
    .login-header .icon { font-size: 2.4rem; }

    /* ── Hero header ── */
    .hero {
        padding: 1.2rem 0 .6rem;
        border-bottom: 1px solid var(--secondary-background-color);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin-bottom: .2rem; }
    .hero p { opacity: .7; margin: 0; }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Result cards ── */
    .result-card {
        padding: 1.2rem;
        border-radius: 10px;
        background: var(--secondary-background-color);
        margin-bottom: 1rem;
    }

    /* ── Metric badge ── */
    .metric-badge {
        display: inline-block;
        padding: .3rem .8rem;
        border-radius: 20px;
        font-size: .85rem;
        font-weight: 600;
        background: rgba(46, 160, 67, .15);
        color: #2ea043;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] [data-testid="stMarkdown"] { font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──
if "token" not in st.session_state:
    st.session_state.token = None
if "processing_result" not in st.session_state:
    st.session_state.processing_result = None


# ── Helper: lightweight preview (3s, 480p) ──
def _make_preview(video_bytes: bytes, ext: str) -> bytes | None:
    tmp_orig_path = None
    tmp_preview_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_orig:
            tmp_orig.write(video_bytes)
            tmp_orig_path = tmp_orig.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_preview:
            tmp_preview_path = tmp_preview.name
        subprocess.run(
            [
                "ffmpeg", "-i", tmp_orig_path,
                "-t", "3",
                "-vf", "scale=-2:480",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-an", "-y",
                tmp_preview_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with open(tmp_preview_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.error("ffmpeg is not installed or not found on PATH.")
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpeg conversion failed (exit {e.returncode}).")
        return None
    finally:
        for p in (tmp_orig_path, tmp_preview_path):
            if p and os.path.exists(p):
                os.unlink(p)


def _render_histogram(
    values: list[float],
    color: str,
    title: str,
    xlabel: str,
    x_min: float | None = None,
    x_max: float | None = None,
) -> bytes:
    """Render a histogram to PNG bytes. Optional x-axis range overrides matplotlib defaults."""
    hist_range = None
    if x_min is not None and x_max is not None and x_max > x_min:
        hist_range = (x_min, x_max)

    fig, ax = plt.subplots()
    ax.hist(values, bins=10, range=hist_range, color=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if hist_range is not None:
        ax.set_xlim(hist_range)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.token is None:
    st.markdown(
        '<div class="login-card">'
        '<div class="login-header">'
        '<div class="icon">🔬</div>'
        "<h2>Cavitation Bubble Analysis</h2>"
        "<p style='opacity:.6'>Sign in to continue</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Sign in", width="stretch", type="primary")
        if submitted:
            try:
                response = requests.post(
                    f"{INTERNAL_API_URL}/token",
                    data={"username": username, "password": password},
                    timeout=30,
                )
                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
            except requests.RequestException:
                st.error("Cannot connect to the API server.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### 🔬 Bubble Analysis")
    st.divider()
    st.markdown(
        "**How to use**\n"
        "1. Upload a video (MP4 / AVI / MOV)\n"
        "2. Preview & click **Process video**\n"
        "3. Download results: annotated video, CSV data, histograms"
    )
    st.divider()
    st.markdown(
        "**Supported formats**\n"
        "- MP4, AVI, MOV\n"
        "- Non-MP4 files are auto-converted for preview"
    )
    st.divider()
    if st.button("Logout", width="stretch"):
        st.session_state.token = None
        st.session_state.processing_result = None
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="hero">'
    "<h1>Cavitation Bubble Tracking</h1>"
    "<p>Upload a video to detect, segment and track cavitation bubbles. "
    "Get annotated video, per-bubble statistics, and histograms.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Upload section ──
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload video")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
    )

with col_preview:
    if uploaded_file is not None:
        st.markdown("#### Preview")
        video_bytes = uploaded_file.getvalue()
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        preview = _make_preview(video_bytes, ext)
        if preview:
            st.video(preview)

# ── Process button ──
if uploaded_file is not None:
    st.divider()
    _left, center, _right = st.columns([1, 2, 1])
    with center:
        process_clicked = st.button(
            "🚀  Process video",
            width="stretch",
            type="primary",
        )

    if process_clicked:
        video_bytes = uploaded_file.getvalue()
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        files = {"file": (uploaded_file.name, video_bytes)}

        progress = st.progress(0, text="Sending video to processing pipeline…")
        try:
            progress.progress(10, text="Processing video — this may take a few minutes…")
            response = requests.post(
                f"{INTERNAL_API_URL}/process_video/",
                files=files,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            progress.progress(90, text="Decoding results…")

            if response.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(response.content))
                video_data = zf.read("output_video.mp4")
                csv_data = zf.read("data.csv")

                speed_hist = zf.read("histogram_speed.png") if "histogram_speed.png" in zf.namelist() else None
                area_hist = zf.read("histogram_area.png") if "histogram_area.png" in zf.namelist() else None

                speeds: list[float] = []
                areas: list[float] = []
                if "histogram_data.json" in zf.namelist():
                    hist_data = json.loads(zf.read("histogram_data.json"))
                    speeds = hist_data.get("speeds", [])
                    areas = hist_data.get("areas", [])

                base_name = os.path.splitext(uploaded_file.name)[0]

                st.session_state.processing_result = {
                    "video_data": video_data,
                    "csv_data": csv_data,
                    "csv_name": f"{base_name}.csv",
                    "video_name": f"{base_name}_processed.mp4",
                    "speed_hist": speed_hist,
                    "speed_hist_name": f"{base_name}_speed.png",
                    "area_hist": area_hist,
                    "area_hist_name": f"{base_name}_area.png",
                    "speeds": speeds,
                    "areas": areas,
                }
                progress.progress(100, text="Done!")
            else:
                progress.empty()
                detail = response.text[:200]
                st.error(f"Processing error ({response.status_code}): {detail}")
        except requests.ConnectionError:
            progress.empty()
            st.error("Lost connection to the API server.")
        except requests.Timeout:
            progress.empty()
            st.error("Processing timed out. Try a shorter video.")
        except requests.RequestException as e:
            progress.empty()
            st.error(f"Network error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.processing_result is not None:
    res = st.session_state.processing_result
    st.divider()
    st.markdown('<span class="metric-badge">✓ Processing complete</span>', unsafe_allow_html=True)

    tab_video, tab_histograms, tab_downloads = st.tabs(
        ["📹 Processed Video", "📊 Histograms", "⬇️ Downloads"]
    )

    # ── Tab: Video ──
    with tab_video:
        st.video(res["video_data"])

    # ── Tab: Histograms ──
    with tab_histograms:
        speeds = res.get("speeds", [])
        areas = res.get("areas", [])
        has_hists = res["speed_hist"] or res["area_hist"]

        if has_hists:
            with st.expander("📐  X-axis range (leave empty for default)", expanded=False):
                st.caption(
                    "Set the X-axis range for each histogram. "
                    "Leave any field empty to fall back to the auto-computed default."
                )
                range_cols = st.columns(4, gap="medium")
                with range_cols[0]:
                    speed_min = st.number_input(
                        "Speed min", value=None, format="%.4f", key="speed_min"
                    )
                with range_cols[1]:
                    speed_max = st.number_input(
                        "Speed max", value=None, format="%.4f", key="speed_max"
                    )
                with range_cols[2]:
                    area_min = st.number_input(
                        "Area min", value=None, format="%.4f", key="area_min"
                    )
                with range_cols[3]:
                    area_max = st.number_input(
                        "Area max", value=None, format="%.4f", key="area_max"
                    )

            speed_img = res["speed_hist"]
            area_img = res["area_hist"]
            if speeds and (speed_min is not None or speed_max is not None):
                speed_img = _render_histogram(
                    speeds,
                    color="blue",
                    title="Speed histogram (top-20 longest-lived)",
                    xlabel="Speed (pixels/frame)",
                    x_min=speed_min,
                    x_max=speed_max,
                )
            if areas and (area_min is not None or area_max is not None):
                area_img = _render_histogram(
                    areas,
                    color="green",
                    title="Area histogram (top-20 longest-lived)",
                    xlabel="Area (pixels^2)",
                    x_min=area_min,
                    x_max=area_max,
                )

            hist_cols = st.columns(2, gap="large")
            if speed_img:
                with hist_cols[0]:
                    st.markdown("**Speed histogram** — top-20 longest-lived bubbles")
                    st.image(speed_img, width="stretch")
            if area_img:
                with hist_cols[1]:
                    st.markdown("**Area histogram** — top-20 longest-lived bubbles")
                    st.image(area_img, width="stretch")
        else:
            st.info("No histograms available (not enough tracked bubbles).")

    # ── Tab: Downloads ──
    with tab_downloads:
        dl_cols = st.columns(4, gap="medium")
        with dl_cols[0]:
            st.download_button(
                "📹  Video",
                data=res["video_data"],
                file_name=res["video_name"],
                width="stretch",
            )
        with dl_cols[1]:
            st.download_button(
                "📄  CSV data",
                data=res["csv_data"],
                file_name=res["csv_name"],
                width="stretch",
            )
        if res["speed_hist"]:
            with dl_cols[2]:
                st.download_button(
                    "📊  Speed hist",
                    data=res["speed_hist"],
                    file_name=res["speed_hist_name"],
                    width="stretch",
                )
        if res["area_hist"]:
            with dl_cols[3]:
                st.download_button(
                    "📊  Area hist",
                    data=res["area_hist"],
                    file_name=res["area_hist_name"],
                    width="stretch",
                )
