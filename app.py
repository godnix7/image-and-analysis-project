from pathlib import Path
from io import BytesIO
import hashlib
import json
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from utils.helpers import load_checkpoint, get_device
from utils.metrics import binary_metrics

st.set_page_config(page_title='Lung Segmentation Studio', page_icon='◉', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#14283B; --muted:#65798B; --teal:#087F83; --line:#D9E3E5; --paper:#F3F6F6; --white:#fff; --red:#C75D57; }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:var(--paper); }
[data-testid="stHeader"] { background:rgba(243,246,246,.9); }
[data-testid="stSidebar"] { background:#E9F0F0; border-right:1px solid var(--line); }
.block-container { max-width:1540px; padding-top:1.6rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
.hero { background:linear-gradient(115deg,#14283B 0%,#214C5D 70%,#087F83 130%); padding:1.55rem 1.8rem; color:white; margin-bottom:1.2rem; }
.hero-kicker { color:#83D4CC; font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.5rem; }
.hero-title { font-family:'Manrope',sans-serif; font-size:2.05rem; line-height:1.13; font-weight:800; letter-spacing:-.04em; color:white; margin:0; }
.hero-copy { color:#D6E2E5; font-size:.94rem; margin-top:.55rem; max-width:850px; }
.section-kicker { color:var(--teal); font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:.8rem 0 .35rem; }
.panel { background:white; border:1px solid var(--line); padding:.9rem; }
.panel-head { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #E8EEEE; padding:.1rem .15rem .65rem; margin-bottom:.65rem; }
.panel-title { font-family:'Manrope',sans-serif; color:var(--ink); font-size:.88rem; font-weight:800; letter-spacing:.01em; }
.tag { color:#087F83; background:#E4F2F0; padding:.22rem .5rem; font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.metric-card { background:white; border:1px solid var(--line); padding:.8rem .95rem; min-height:76px; }
.metric-label { color:#718393; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.metric-value { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.18rem; font-weight:800; margin-top:.25rem; }
.note { color:#6C7E8C; font-size:.76rem; line-height:1.5; }
.status { display:flex;align-items:center;gap:.5rem;color:#3E5965;font-size:.82rem;font-weight:700; }
.dot { width:9px;height:9px;border-radius:50%;background:#168C73;display:inline-block; }
div.stButton>button[kind="primary"] { background:var(--teal);border:1px solid var(--teal);border-radius:3px;font-weight:700; }
div.stButton>button { border-radius:3px;font-weight:600; }
[data-testid="stFileUploader"] section { background:white;border:1px dashed #A9BEC0;border-radius:3px; }
[data-testid="stMetric"] { background:white;border:1px solid var(--line);padding:.65rem .8rem; }
iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"] { height:min(760px, calc(50vw - 23px)) !important; }
@media (max-width:640px) { iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"] { height:min(640px, calc(100vw - 32px)) !important; } }
code { color:#087F83 !important; }
</style>
''', unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DEVICE = get_device()

with st.sidebar:
    st.markdown('<div class="section-kicker">MEDICAL AI / RESEARCH TOOL</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:1.24rem;line-height:1.12;margin:.2rem 0">Lung Segmentation<br>Studio</h2>', unsafe_allow_html=True)
    checkpoint = st.text_input('Checkpoint', 'checkpoints/best.pt')
    model_ready = Path(checkpoint).is_file()
    st.markdown(f'<div class="status"><span class="dot" style="background:{"#168C73" if model_ready else "#C58A2B"}"></span>{"Trained model ready" if model_ready else "Checkpoint not found"}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">HOW TO CORRECT</div><div class="note"><b style="color:#087F83">Positive click</b> inside a missed lung area.<br><b style="color:#C75D57">Negative click</b> on an area wrongly included.<br><br>Click the X-ray preview, choose a point type, then apply it. Undo and reset are available.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">MODEL</div><div class="note">Four-channel U-Net<br>256 × 256 inference<br>CUDA when available</div>', unsafe_allow_html=True)

st.markdown('''<div class="hero"><div class="hero-kicker">INTERACTIVE CHEST X-RAY SEGMENTATION</div>
<div class="hero-title">Refine the lung outline.<br>See each change.</div>
<div class="hero-copy">Start from a held-out benchmark case or bring your own radiograph. Add positive and negative clicks, compare the mask, then export the result.</div></div>''', unsafe_allow_html=True)

if not model_ready:
    st.error('The trained checkpoint is missing. Expected file: checkpoints/best.pt')
    st.stop()

@st.cache_resource
def model_load(path, device_name):
    return load_checkpoint(path, torch.device(device_name))

try:
    model, ckpt = model_load(str(Path(checkpoint).resolve()), str(DEVICE))
    model.eval()
    size = int(ckpt['config']['size'])
except Exception as exc:
    st.error(f'Could not load checkpoint: {exc}')
    st.stop()

# Use one actual held-out NLM case as a ready-to-inspect demo, including its reference mask.
config_path = ROOT / 'results' / 'config.json'
metrics_path = ROOT / 'results' / 'evaluation' / 'metrics.json'
demo_root = ROOT / 'data' / 'MontgomerySet' / 'CXR_png'
demo_cases = []
if config_path.is_file() and metrics_path.is_file() and demo_root.is_dir():
    try:
        evaluated = json.loads(metrics_path.read_text(encoding='utf-8'))
        demo_cases = sorted({Path(row['image']).name for row in evaluated.get('per_image', []) if (demo_root / Path(row['image']).name).is_file()})
    except (ValueError, KeyError, OSError):
        demo_cases = []

source_options = ['Upload my own image']
if demo_cases:
    source_options.append('Try a sample image')
source = st.radio('Start here', source_options, horizontal=True)
up = None
gt = None
case_name = None
if source == 'Try a sample image':
    case_name = st.selectbox('Held-out NLM case', demo_cases, label_visibility='collapsed')
    image_path = demo_root / case_name
    raw_bytes = image_path.read_bytes()
    raw = Image.open(BytesIO(raw_bytes)).convert('L')
    gt_path = demo_root / 'FusedMasks' / case_name
    if gt_path.is_file():
        gt = Image.open(gt_path).convert('L')
    st.caption('Sample chest X-ray from the NLM Montgomery collection. Its reference mask is available for comparison.')
else:
    up = st.file_uploader('Upload a chest X-ray image', type=['png','jpg','jpeg','bmp','tif','tiff'], help='Supported formats: PNG, JPG, JPEG, BMP, and TIFF.')
    if up is None:
        st.info('Choose an image above, or select “Try a sample image” to explore an example.')
        st.stop()
    raw_bytes = up.getvalue()
    try:
        raw = Image.open(BytesIO(raw_bytes)).convert('L')
    except Exception as exc:
        st.error(f'This image could not be opened: {exc}')
        st.stop()

orig_size = raw.size
image_id = hashlib.sha1(raw_bytes + str(case_name).encode()).hexdigest()
resized = raw.resize((size, size), Image.Resampling.BILINEAR)
arr = np.asarray(resized, dtype=np.float32) / 255.0
x = torch.from_numpy(arr[None, None]).to(DEVICE)
if st.session_state.get('image_id') != image_id:
    st.session_state.image_id = image_id
    st.session_state.pos = torch.zeros_like(x)
    st.session_state.neg = torch.zeros_like(x)
    st.session_state.prev = torch.zeros_like(x)
    st.session_state.clicks = []
    st.session_state.pending_click = None
    st.session_state.pending_event_id = None
    st.session_state.last_click_event = None

with torch.no_grad():
    logits = model(torch.cat([x, st.session_state.pos, st.session_state.neg, st.session_state.prev], dim=1))
    prob = torch.sigmoid(logits)
    mask = (prob[0, 0].cpu().numpy() >= .5).astype(np.uint8)

full_mask = np.asarray(Image.fromarray(mask * 255).resize(orig_size, Image.Resampling.NEAREST))
overlay = np.asarray(raw.convert('RGB')).copy()
green = np.zeros_like(overlay); green[:, :, 1] = 190
covered = full_mask > 0
overlay[covered] = (.58 * overlay[covered] + .42 * green[covered]).astype(np.uint8)
for click in st.session_state.clicks:
    px = int(round(click['x'] * orig_size[0] / size)); py = int(round(click['y'] * orig_size[1] / size))
    d = max(7, round(min(orig_size) / 100))
    colour = (18, 160, 127) if click['type'] == 'positive' else (205, 75, 75)
    # Draw on the persistent array so click markers survive the next loop iteration.
    marker = Image.fromarray(overlay); ImageDraw.Draw(marker).ellipse((px-d, py-d, px+d, py+d), fill=colour, outline='white', width=max(2, d//3)); overlay = np.asarray(marker).copy()

metrics = binary_metrics(mask, np.asarray(gt.resize((size, size), Image.Resampling.NEAREST)) > 0) if gt is not None else None
if metrics:
    values = [('DICE', f"{metrics['dice']:.3f}"), ('IOU', f"{metrics['iou']:.3f}"), ('CLICKS', str(len(st.session_state.clicks))), ('MODEL', f"{DEVICE.type.upper()} · {size}px")]
else:
    values = [('CLICKS', str(len(st.session_state.clicks))), ('MODEL', f"U-Net · {size}px"), ('DEVICE', DEVICE.type.upper()), ('IMAGE', f"{orig_size[0]} × {orig_size[1]}")]
cols = st.columns(4)
for col, (label, value) in zip(cols, values):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
left, right = st.columns([1.08, 1], gap='medium')
with left:
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-title">1 / Correct the prediction</span><span class="tag">Updates on click</span></div></div>', unsafe_allow_html=True)
    st.caption('Choose what needs fixing, then click the image. The model updates right away.')
    mode = st.radio('Click action', ['Add missed lung area', 'Remove wrongly included area'], horizontal=True)
    display = Image.fromarray(overlay).resize((640, 640), Image.Resampling.BILINEAR)
    coords = streamlit_image_coordinates(display, key=f'click_{image_id}', width='stretch')
    if coords and coords.get('x') is not None and coords.get('unix_time') != st.session_state.last_click_event:
        px = int(coords['x'] * size / coords['width'])
        py = int(coords['y'] * size / coords['height'])
        key = 'pos' if mode.startswith('Add') else 'neg'
        yy, xx = torch.meshgrid(torch.arange(size, device=DEVICE), torch.arange(size, device=DEVICE), indexing='ij')
        g = torch.exp(-((xx-px)**2 + (yy-py)**2) / (2 * 12**2))[None, None]
        st.session_state[key] = torch.maximum(st.session_state[key], g)
        st.session_state.clicks.append({'x': px, 'y': py, 'type': key})
        st.session_state.last_click_event = coords.get('unix_time')
        with torch.no_grad():
            st.session_state.prev = torch.sigmoid(model(torch.cat([x, st.session_state.pos, st.session_state.neg, st.session_state.prev], 1)))
        st.rerun()
    undo_col, clear_col = st.columns(2)
    with undo_col:
        if st.button('Undo last click', use_container_width=True, disabled=not st.session_state.clicks, help='Remove the most recent correction and rebuild the prediction.'):
            st.session_state.clicks.pop(); st.session_state.pos.zero_(); st.session_state.neg.zero_(); st.session_state.prev.zero_()
            yy, xx = torch.meshgrid(torch.arange(size, device=DEVICE), torch.arange(size, device=DEVICE), indexing='ij')
            for point in st.session_state.clicks:
                g = torch.exp(-((xx-point['x'])**2 + (yy-point['y'])**2) / (2 * 12**2))[None, None]
                key = point['type']; st.session_state[key] = torch.maximum(st.session_state[key], g)
                with torch.no_grad():
                    st.session_state.prev = torch.sigmoid(model(torch.cat([x, st.session_state.pos, st.session_state.neg, st.session_state.prev], 1)))
            st.rerun()
    with clear_col:
        if st.button('Clear all clicks', use_container_width=True, disabled=not st.session_state.clicks, help='Remove every correction and return to the original prediction.'):
            st.session_state.pos.zero_(); st.session_state.neg.zero_(); st.session_state.prev.zero_(); st.session_state.clicks = []; st.session_state.pending_click = None; st.session_state.pending_event_id = None; st.session_state.last_click_event = None; st.rerun()

with right:
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-title">2 / Inspect the mask</span><span class="tag">Live prediction</span></div></div>', unsafe_allow_html=True)
    views = ['Overlay', 'Binary mask'] + (['Reference mask'] if gt is not None else [])
    view = st.segmented_control('Result view', views, default='Overlay', label_visibility='collapsed') if hasattr(st, 'segmented_control') else st.radio('Result view', views, horizontal=True, label_visibility='collapsed')
    if view == 'Binary mask':
        st.image(full_mask, use_container_width=True, clamp=True)
    elif view == 'Reference mask' and gt is not None:
        st.image(np.asarray(gt), use_container_width=True, clamp=True)
    else:
        st.image(overlay, use_container_width=True)
    if metrics:
        st.markdown(f'<div class="note">Benchmark comparison against the provided fused expert reference. Current Dice {metrics["dice"]:.3f} / IoU {metrics["iou"]:.3f}. Results are for this image only.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">No reference mask was supplied, so this view shows the model output without quality scores.</div>', unsafe_allow_html=True)

with st.expander('Compare with a reference mask'):
    if gt is not None:
        st.caption('The benchmark reference mask is already loaded for this example.')
    else:
        gtup = st.file_uploader('Optional reference mask', type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'], key='ground_truth')
        if gtup:
            gt_user = Image.open(gtup).convert('L').resize((size, size), Image.Resampling.NEAREST)
            user_metrics = binary_metrics(mask, np.asarray(gt_user) > 0)
            st.write({**user_metrics, 'refinement_clicks': len(st.session_state.clicks)})

st.markdown('<div class="section-kicker">EXPORT</div>', unsafe_allow_html=True)
mask_buf = BytesIO(); Image.fromarray(mask * 255).resize(orig_size, Image.Resampling.NEAREST).save(mask_buf, format='PNG')
overlay_buf = BytesIO(); Image.fromarray(overlay).save(overlay_buf, format='PNG')
d1, d2, d3 = st.columns([1, 1, 2])
with d1: st.download_button('Download mask', mask_buf.getvalue(), 'lung_mask.png', 'image/png', use_container_width=True)
with d2: st.download_button('Download overlay', overlay_buf.getvalue(), 'lung_overlay.png', 'image/png', use_container_width=True)
with d3: st.caption('PNG exports preserve the original radiograph dimensions.')
st.caption('Research prototype for lung-field segmentation. Not validated for clinical use.')
