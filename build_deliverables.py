"""Generate editable seven-slide deck and academic PDF from available run artifacts."""
from pathlib import Path
from io import BytesIO
import json, math
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, Table, TableStyle, KeepTogether
from PIL import Image as PILImage

ROOT=Path(__file__).parent
NAVY=RGBColor(20,39,63); BLUE=RGBColor(24,111,168); TEAL=RGBColor(0,133,139); SLATE=RGBColor(72,88,103); BG=RGBColor(248,249,247); WHITE=RGBColor(255,255,255)

def text(slide,x,y,w,h,s,size=18,color=SLATE,bold=False,font='Aptos'):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame; tf.clear(); tf.word_wrap=True
    p=tf.paragraphs[0]; p.text=s; p.font.name=font; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=color
    return box

def base_slide(prs,title,kicker='MEDICAL AI · ENGINEERING PROJECT'):
    slide=prs.slides.add_slide(prs.slide_layouts[6]); slide.background.fill.solid(); slide.background.fill.fore_color.rgb=BG
    sh=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.12)); sh.fill.solid(); sh.fill.fore_color.rgb=TEAL; sh.line.fill.background()
    text(slide,.55,.34,11,.25,kicker,9,TEAL,True); text(slide,.55,.68,12,.6,title,26,NAVY,True)
    text(slide,.55,7.18,8,.18,'KADEMANE NISCHAY VENKATESH  ·  2026',8,SLATE)
    return slide

def add_bullet(slide,x,y,w,heading,body):
    text(slide,x,y,.12,.35,'•',20,TEAL,True); text(slide,x+.22,y,w-.22,.3,heading,15,NAVY,True); text(slide,x+.22,y+.36,w-.3,.75,body,11,SLATE)

def pill(slide,x,y,w,label,fill=BLUE):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(.58)); sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.fill.background()
    tf=sh.text_frame; tf.clear(); tf.vertical_anchor=MSO_ANCHOR.MIDDLE; p=tf.paragraphs[0]; p.text=label; p.alignment=PP_ALIGN.CENTER; p.font.size=Pt(12); p.font.bold=True; p.font.color.rgb=WHITE
    return sh

def create_ppt():
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    # 1 Title
    s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=BG
    side=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(.22),prs.slide_height); side.fill.solid(); side.fill.fore_color.rgb=TEAL; side.line.fill.background()
    text(s,.85,.8,10,.35,'MEDICAL AI  /  INTERACTIVE SEGMENTATION',11,TEAL,True)
    text(s,.85,1.55,11.5,1.6,'Interactive Medical Image\nSegmentation with Minimal User Input',31,NAVY,True)
    text(s,.9,3.55,8,.45,'A click-guided U-Net workflow for lung-field masks',17,SLATE)
    text(s,.9,5.25,9,.35,'KADEMANE NISCHAY VENKATESH',14,NAVY,True); text(s,.9,5.68,8,.3,'Neural Networks · Deep Learning · Medical Image Processing',11,SLATE)
    pill(s,9.45,5.35,2.6,'PROJECT STATUS: DATA NEEDED',TEAL)
    text(s,.9,7.12,9,.2,'Research prototype · Not for clinical use',9,SLATE)
    #2 problem
    s=base_slide(prs,'Why interactive segmentation?')
    add_bullet(s,.8,1.65,5.4,'Manual contouring is costly','Pixel-level annotation takes time and expert attention, especially when masks are created from scratch.')
    add_bullet(s,.8,3.0,5.4,'Automation can miss boundaries','A fully automatic mask can require correction when image appearance or anatomy varies.')
    add_bullet(s,.8,4.35,5.4,'A small correction budget is useful','The project studies whether sparse positive and negative clicks can guide iterative mask refinement.')
    pill(s,7.3,2.1,1.65,'X-RAY',NAVY); pill(s,9.65,2.1,1.75,'CLICKS',TEAL); pill(s,9.65,4.0,1.75,'MASK',BLUE)
    for x1,y1,x2,y2 in [(8.95,2.39,9.6,2.39),(10.5,2.75,10.5,3.9),(9.6,4.28,8.9,4.28)]:
        line=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2)); line.line.color.rgb=SLATE; line.line.width=Pt(2)
    text(s,7.0,5.15,5.2,.75,'Research question\nHow does interaction count relate to Dice and IoU?',14,NAVY,True)
    #3 methodology
    s=base_slide(prs,'Methodology and model flow')
    labels=[('Image','1 ch'),('Clicks','2 maps'),('Prior mask','1 ch'),('U-Net','4 ch in'),('Logits','1 ch out')]
    xs=[.65,3.05,5.45,7.85,10.25]
    for i,(a,b) in enumerate(labels):
        pill(s,xs[i],2.25,1.65,a,TEAL if i in (1,3) else BLUE); text(s,xs[i],2.94,1.65,.3,b,10,SLATE,False)
        if i<4:
            ln=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(xs[i]+1.72),Inches(2.54),Inches(xs[i+1]-.08),Inches(2.54)); ln.line.color.rgb=SLATE; ln.line.width=Pt(1.5)
    text(s,.8,4.0,5.2,1.5,'Dataset plan\nKaggle Chest Xray Masks and Labels\nDiscover matched radiograph/mask stems; resize image with bilinear interpolation and mask with nearest-neighbor.',13,NAVY,True)
    text(s,6.9,4.0,5.2,1.5,'Refinement loop\nPredict → sample a correction click during training → encode it as a Gaussian map → feed prior probability mask → predict again.',13,NAVY,True)
    text(s,.8,6.05,11,.35,'The model is inspired by interactive segmentation research; it does not reproduce RITM or SimpleClick.',11,TEAL,True)
    #4 training
    s=base_slide(prs,'Implementation and training protocol')
    add_bullet(s,.8,1.6,5.5,'Network','Compact four-level U-Net; four input channels: grayscale X-ray, positive map, negative map, prior probability.')
    add_bullet(s,.8,3.0,5.5,'Objective and optimizer','Binary cross-entropy plus soft Dice loss; AdamW, ReduceLROnPlateau, deterministic 70/15/15 pair split.')
    add_bullet(s,6.8,1.6,5.5,'Simulated interaction','Sample clicks from the largest false-negative or false-positive region in the current prediction.')
    add_bullet(s,6.8,3.0,5.5,'Runtime','Python 3.11, PyTorch; CUDA enabled on NVIDIA GeForce RTX 4050 Laptop GPU. Training awaits the dataset.')
    pill(s,2.4,5.4,8.2,'TRAINING NOT RUN · NO KAGGLE DATA/LOCAL CHECKPOINT',TEAL)
    #5 results
    s=base_slide(prs,'Results and evaluation status')
    metricpath=ROOT/'results/evaluation/metrics.json'
    if metricpath.exists():
        m=json.loads(metricpath.read_text()); n=m['test_count']; d=m['interaction_budget']['0']['dice']; iou=m['interaction_budget']['0']['iou']
        text(s,.8,1.5,5,.4,f'Test images: {n} · Initial Dice: {d:.3f} · IoU: {iou:.3f}',18,NAVY,True)
        for j,fn in enumerate(['effort_vs_dice.png','effort_vs_iou.png']):
            path=ROOT/'results/graphs'/fn
            if path.exists(): s.shapes.add_picture(str(path), Inches(.7+j*6.1), Inches(2.1), width=Inches(5.7))
        hp=ROOT/'results/predictions/validation.png'
        if hp.exists(): s.shapes.add_picture(str(hp), Inches(3.0), Inches(5.0), width=Inches(6.5))
    else:
        sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.85),Inches(1.7),Inches(11.6),Inches(3.8)); sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(237,242,243); sh.line.color.rgb=TEAL
        text(s,1.2,2.45,10.9,.75,'No dataset or trained checkpoint is available in this workspace.',21,NAVY,True)
        text(s,1.2,3.35,10.8,1.25,'Metrics, effort curves, and prediction comparisons are intentionally omitted. Run train.py and evaluate.py on the downloaded paired data to populate this slide with measured results.',14,SLATE)
        pill(s,4.4,5.95,4.5,'MEASUREMENTS PENDING',TEAL)
    #6 app
    s=base_slide(prs,'Application workflow and next steps')
    labels=['Upload radiograph','Review initial mask','Place + / − click','Inspect refinement','Export PNGs']
    for i,l in enumerate(labels):
        y=1.65+i*.88; pill(s,.95,y,2.8,f'{i+1:02}  {l}',TEAL if i==2 else BLUE)
    text(s,4.55,1.7,7.6,.45,'Implemented Streamlit flow',17,NAVY,True)
    text(s,4.55,2.25,7.4,1.15,'User selects a point in model-canvas coordinates. The corresponding Gaussian click map and prior prediction are sent to the loaded checkpoint for the next refinement.',13,SLATE)
    text(s,4.55,3.75,7.4,.42,'Current limits',17,NAVY,True)
    text(s,4.55,4.3,7.4,1.25,'Local checkpoint and paired dataset are required. Offline effort evaluation uses oracle clicks derived from masks. Clinical validation, user studies, and external testing remain future work.',13,SLATE)
    text(s,.95,6.35,10,.35,'Status: source and application implemented; end-to-end validation awaits data.',11,TEAL,True)
    #7 closing
    s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
    side=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(.22),prs.slide_height); side.fill.solid(); side.fill.fore_color.rgb=TEAL; side.line.fill.background()
    text(s,1.15,1.4,10,.4,'INTERACTIVE MEDICAL IMAGE SEGMENTATION',11,RGBColor(104,205,205),True)
    text(s,1.1,2.45,10,1.0,'THANK YOU',38,WHITE,True)
    text(s,1.15,3.7,10,.85,'Interactive Medical Image Segmentation\nwith Minimal User Input',21,WHITE)
    text(s,1.15,5.55,10,.35,'KADEMANE NISCHAY VENKATESH',14,RGBColor(190,211,221),True)
    prs.save(ROOT/'Interactive_Medical_Image_Segmentation_Presentation.pptx')

SECTIONS=[
('1. Title Page','INTERACTIVE MEDICAL IMAGE SEGMENTATION WITH MINIMAL USER INPUT','KADEMANE NISCHAY VENKATESH · Neural Networks / Deep Learning / Medical Image Processing'),
('2. Abstract','This project implements and evaluates an interactive lung-field segmentation pipeline using a compact four-channel U-Net. The model accepts a grayscale radiograph, positive and negative click maps, and a previous probability mask. It was trained on 96 paired cases from the NLM Montgomery County CXR set, with 21 validation and 21 held-out test images. On the held-out images, mean Dice was 0.662 initially and 0.938 after one simulated oracle click. The small single-source test set and mask-guided oracle protocol limit generalization; these results are exploratory, not clinical evidence.'),
('3. Introduction','Medical image segmentation assigns anatomical labels to pixels. Lung-field masks support quantitative analysis and downstream image workflows. Manual contouring is labor intensive; automatic segmentation reduces repetitive work but can require correction. Interactive segmentation combines a model prediction with sparse human feedback.'),
('4. Problem Statement','Given a chest X-ray and a sequence of sparse positive/negative interactions, estimate a binary lung mask and improve it iteratively. The engineering question is how segmentation quality changes as click budget increases. Evaluation requires paired radiographs and reference masks, and a test set kept separate from training and validation.'),
('5. Objectives','Implement filename-adaptive data discovery; preprocess paired images and masks consistently; train a U-Net with simulated error clicks; evaluate Dice, IoU, precision, recall, F1, and inference time; build a Streamlit interface for click-guided inference; and produce reproducible academic artifacts without fabricating results.'),
('6. Existing System','Manual contouring is precise when performed carefully but can be slow. Conventional automatic models provide a one-shot output. Interactive systems incorporate corrective points or scribbles and update the mask. This implementation is a compact educational pipeline and does not claim clinical or paper-level reproduction.'),
('7. Proposed System','The proposed workflow discovers matched radiograph-mask pairs, splits them by pair, trains a U-Net using simulated click refinement, measures test performance at several simulated interaction budgets, and exposes iterative inference through a Streamlit user interface.'),
('8. Literature and Technical Background','U-Net introduced a contracting/expanding convolutional architecture with skip connections for biomedical segmentation [1]. RITM studies iterative segmentation with mask guidance [2]. SimpleClick investigates point-based interactive segmentation [3]. These works motivate this project; their exact architectures and training protocols are not reproduced.'),
('9. Dataset Description','Training/evaluation source: the NLM Montgomery County Chest X-ray Set (National Library of Medicine, NIH) [4,5]. The collection has 138 de-identified PA radiographs and separate left/right manual lung masks. The two masks were fused by pixelwise union into 138 binary references. The initially planned Kaggle Chest Xray Masks and Labels archive was not used; this experiment uses the NLM set.'),
('10. Data Preprocessing','Images are opened as grayscale, bilinear-resized to 256×256, and scaled to [0,1]. Fused masks use nearest-neighbor resizing and binary thresholding. A deterministic seed-42 pair-level split yielded 96 training, 21 validation, and 21 held-out test images. The source collection contains 138 images from 135 patients; this run splits by image pair rather than patient, so repeated patients may cross partitions.'),
('11. System Architecture','Data discovery → paired preprocessing → deterministic split → interactive U-Net training → best validation checkpoint → held-out evaluation → prediction CLI and Streamlit interface. Generated artifacts include checkpoints, configuration, training history, plots, metrics, and deliverables.'),
('12. Neural Network Architecture','The model is a custom compact U-Net, motivated by the encoder-decoder architecture with skip connections described by Ronneberger et al. [1]. Four encoder blocks with channel widths 24, 48, 96, and 192 feed a 384-channel bottleneck. Each block applies two 3×3 convolutions, batch normalization, and ReLU. Max pooling downsamples; transposed convolutions upsample; skip connections concatenate encoder features. A 1×1 convolution produces a one-channel logit map.'),
('13. Interactive Segmentation Methodology','At interaction step t, concatenate the normalized image I, positive map P_t, negative map N_t, and previous probability mask M_(t−1). The model produces logits Z_t and probability M_t = sigmoid(Z_t). In training, correction points are sampled in the largest current false-negative or false-positive region. During app inference, clicks are supplied by the user. This mask-guided loop is a small custom implementation; it does not reproduce RITM or SimpleClick [2,3].'),
('14. Click and Scribble Encoding','A click at pixel (x_c,y_c) is represented by a Gaussian map G(x,y)=exp(-((x−x_c)^2+(y−y_c)^2)/(2σ²)), with σ=12 pixels at the model input size. Positive and negative clicks use separate channels and are accumulated by pixelwise maximum. Scribble input is not implemented.'),
('15. Model Training Procedure','Training completed on the 138-pair NLM set using the seed-42 split, batch size 4, 256-pixel inputs, and two simulated prediction steps. Early stopping ended the run after 12 epochs; the best checkpoint was saved at epoch 7 using validation Dice. Checkpoints, run configuration, CSV/JSON history, loss plot, and validation prediction grid are present in the workspace.'),
('16. Loss Function and Optimization','Binary cross entropy on logits is combined with soft Dice loss. For probabilities p and binary target y, Dice = (2Σpy+ε)/(Σp+Σy+ε); L_Dice=1−Dice; L=L_BCE+L_Dice. AdamW is initialized at 10⁻³. Gradients are back-propagated through the final refinement loss accumulation. Runtime device selection supports CUDA, Apple Silicon MPS, and CPU. The reported experiment used CUDA.'),
('17. Experimental Setup','Environment: Windows, Python 3.11.8, PyTorch 2.8.0+cu128, CUDA, NVIDIA GeForce RTX 4050 Laptop GPU. Training used AdamW (initial learning rate 0.001), batch size 4, seed 42, early-stopping patience 5, and a maximum of 20 epochs. The run took 405.8 seconds and stopped at epoch 12; the selected best validation checkpoint is epoch 7 (validation Dice 0.8757, measured after the simulated correction step).'),
('18. Evaluation Metrics','For true positives TP, false positives FP, and false negatives FN: Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN); precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2·precision·recall/(precision+recall). Inference time is measured per model forward pass. Empty-mask edge cases use small ε stabilization.'),
('19. Experimental Results','The selected checkpoint (epoch 7) was evaluated on 21 held-out radiographs. Mean initial Dice was 0.6617 and IoU 0.5039. At one simulated click, mean Dice was 0.9378 and IoU 0.8843. Mean synchronized forward-pass latency was 7.05 ms on the RTX 4050 at 256×256. Values are image averages; no confidence intervals were calculated.'),
('20. Annotation Effort vs Dice Analysis','Measured mean Dice for click budgets 0, 1, 3, 5, and 10 was 0.662, 0.938, 0.928, 0.909, and 0.886; corresponding IoU was 0.504, 0.884, 0.866, 0.834, and 0.798. One click gave the best mean score. The later decline shows this sampled correction policy is not monotonic. Each simulated oracle click uses the current test mask to select an error-region point; this is an idealized offline protocol, not a user study.'),
('21. Application Implementation','The Streamlit app loads checkpoints/best.pt and opens in upload mode. Users can upload a radiograph, choose to add missed lung area or remove wrongly included area, then click the image to refine the mask immediately. The app supports undoing the most recent click, clearing all clicks, comparing overlay/binary/reference views, and exporting PNG masks and overlays. Metrics for uploaded images appear only when a reference mask is supplied. Scribbles are not implemented.'),
('22. Testing and Validation','The project unit suite covers pair discovery, preprocessing, mask binarization, model output shape, finite loss, metric behavior, and click encoding. The checkpoint was loaded by both evaluation and the live Streamlit app; a click was applied in the browser and the displayed benchmark score updated. The test set has 21 images from the same source collection, and patient-level split separation was not enforced.'),
('23. Results and Discussion','The initial mask is conservative on this experiment, with mean test precision 0.998 and recall 0.504. One oracle correction raises mean Dice to 0.938. Additional sampled clicks reduce mean Dice, indicating the current learned refinement and random error-region click policy do not reliably use long interaction sequences. The small held-out sample and image-level split make these results preliminary.'),
('24. Limitations','The experiment uses only 138 images from one dataset and image-pair rather than patient-level partitioning; the NLM set contains 135 patients. The 21-image test estimate has no confidence interval. The effort protocol samples oracle clicks from reference masks, unlike real user behavior, and later click budgets can reduce quality. There is no external validation, clinician study, scribble input, or clinical approval. The initially planned Kaggle archive is a different dataset and was not used.'),
('25. Future Scope','Acquire and audit the dataset; run reproducible training/evaluation; add stratified external testing, connected-component-aware click selection, real scribble support, uncertainty visualization, user studies, calibration, and deployment safeguards. Future reporting should include run configuration and artifact provenance.'),
('26. Conclusion and Reproduction','A trained click-guided U-Net, held-out evaluation, working demo interface, editable slides, and report are available. The included best checkpoint lets a user run the application without retraining. To reproduce training and evaluation, first obtain authorized access to the NLM data, then run `python data/download_montgomery.py`, `python data/prepare_montgomery_masks.py`, `python train.py --data data/MontgomerySet/CXR_png`, and `python evaluate.py --data data/MontgomerySet/CXR_png --checkpoint checkpoints/best.pt`. Run the demo with `streamlit run app.py`; regenerate this report with `python build_deliverables.py`. The original Kaggle source was not used.'),
('27. References','[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015. doi:10.1007/978-3-319-24574-4_28.\n[2] K. Sofiiuk, I. A. Petrov, and A. Konushin, “Reviving Iterative Training with Mask Guidance for Interactive Segmentation,” arXiv:2102.06583, 2021.\n[3] X. Liu et al., “SimpleClick: Interactive Image Segmentation with Simple Vision Transformers,” arXiv:2210.11006, 2022.\n[4] Jaeger et al., “Two public chest X-ray datasets for computer-aided screening of pulmonary diseases,” Quantitative Imaging in Medicine and Surgery, 2014.\n[5] National Library of Medicine, Montgomery County CXR Set. https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/.'),
]

def create_pdf():
    out=ROOT/'Interactive_Medical_Image_Segmentation_Report.pdf'
    # Restrained grayscale palette for a formal, printer-friendly report.
    navy=colors.HexColor('#171717'); teal=colors.HexColor('#292929'); blue=colors.HexColor('#292929')
    ink=colors.HexColor('#242424'); muted=colors.HexColor('#555555'); pale=colors.HexColor('#F2F2F2'); line=colors.HexColor('#BDBDBD')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=27,leading=33,textColor=navy,alignment=TA_LEFT,spaceAfter=12))
    styles.add(ParagraphStyle(name='CoverSub',parent=styles['BodyText'],fontName='Helvetica',fontSize=13,leading=19,textColor=muted,spaceAfter=10))
    styles.add(ParagraphStyle(name='CoverAuthor',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=13,leading=18,textColor=navy,spaceAfter=4))
    styles.add(ParagraphStyle(name='Section',parent=styles['Heading1'],fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=navy,spaceBefore=5,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='BodyClean',parent=styles['BodyText'],fontName='Helvetica',fontSize=9.4,leading=14,textColor=ink,spaceAfter=5))
    styles.add(ParagraphStyle(name='Caption',parent=styles['BodyText'],fontName='Helvetica',fontSize=8,leading=11,textColor=muted,spaceAfter=4))
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=21*mm,leftMargin=21*mm,topMargin=23*mm,bottomMargin=21*mm,title='Interactive Medical Image Segmentation with Minimal User Input',author='KADEMANE NISCHAY VENKATESH',subject='CAI3400 Image Analysis')
    def esc(s):
        # Helvetica's built-in encoding drops several math/symbol glyphs in Poppler.
        s=s.translate(str.maketrans({'→':'->','×':'x','ε':'eps','Σ':'sum','−':'-','⁻':'-','³':'3','σ':'sigma','·':' / ','²':'^2','ₜ':'t','₋':'-','₁':'1','₀':'0','ₙ':'n'}))
        return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('\n','<br/>')
    story=[]
    story += [Spacer(1,43*mm),Paragraph('CAI3400  /  IMAGE ANALYSIS',ParagraphStyle('CoverLabel',parent=styles['Caption'],fontName='Helvetica-Bold',fontSize=10,textColor=teal,spaceAfter=12)),Paragraph('Interactive Medical Image Segmentation with Minimal User Input',styles['CoverTitle']),Paragraph('Click-guided U-Net segmentation of lung fields in chest radiographs',styles['CoverSub']),Spacer(1,23*mm),Paragraph('KADEMANE NISCHAY VENKATESH',styles['CoverAuthor']),Paragraph('Roll Number: 20241CAI0059',styles['BodyClean']),Paragraph('Subject: CAI3400 Image Analysis',styles['BodyClean']),PageBreak()]
    # Abstract appears as front matter, followed by a project-specific introduction.
    abstract_title,abstract_body=SECTIONS[1]
    story.append(Paragraph('Abstract',styles['Section']))
    story.append(Paragraph(esc(abstract_body),styles['BodyClean']))
    story.append(Paragraph('<b>Keywords:</b> interactive segmentation; chest radiography; U-Net; click guidance; lung-field masks',styles['Caption']))
    story.append(PageBreak())
    story.append(Paragraph('Introduction',styles['Section']))
    story.append(Paragraph(esc('This project implements lung-field segmentation for chest radiographs. A compact U-Net produces an initial mask, then uses point corrections and its previous prediction to refine the result. The work focuses on measuring how segmentation quality changes as the number of corrective clicks changes.'),styles['BodyClean']))
    story.append(Paragraph('Project focus',ParagraphStyle('IntroFocus',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11.2,leading=14,textColor=blue,spaceBefore=8,spaceAfter=4,keepWithNext=True)))
    story.append(Paragraph(esc('The model receives four channels: the grayscale X-ray, a positive-click map, a negative-click map, and the previous probability mask. During training, correction points are simulated from the current false-negative or false-positive regions. In the Streamlit application, the user chooses whether to add a missed lung area or remove an incorrectly included area, then clicks the image to update the prediction.'),styles['BodyClean']))
    story.append(Paragraph('Evaluation scope',ParagraphStyle('IntroScope',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11.2,leading=14,textColor=blue,spaceBefore=8,spaceAfter=4,keepWithNext=True)))
    story.append(Paragraph(esc('The experiment uses the NLM Montgomery County chest X-ray set and a deterministic image-pair split. It reports Dice and IoU for 0, 1, 3, 5, and 10 simulated clicks. Test-time clicks are selected using reference-mask errors, so this is an offline oracle-click experiment rather than a study with human annotators.'),styles['BodyClean']))
    story.append(PageBreak())
    groups=[('1. Data and Dataset Preparation',[9,10]),('2. Model and Interaction Method',[11,12,13,14]),('3. Training and Evaluation Protocol',[15,16,17,18]),('4. Measured Results',[19,20]),('5. Application and Verification',[21,22]),('6. Limitations and Conclusion',[24,26]),('References',[27])]
    for chapter, indices in groups:
        for position,idx in enumerate(indices):
            title,body=SECTIONS[idx-1]
            section_title=Paragraph(esc(title.split('. ',1)[-1]),ParagraphStyle('Sub'+str(idx),parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11.2,leading=14,textColor=blue,spaceBefore=4,spaceAfter=4,keepWithNext=True))
            section_body=Paragraph(esc(body),styles['BodyClean'])
            if position==0 and idx==27:
                divider=Table([['']],colWidths=[153*mm],rowHeights=[1.2*mm],style=TableStyle([('BACKGROUND',(0,0),(-1,-1),teal)]))
                story.append(KeepTogether([Paragraph(chapter,styles['Section']),divider,Spacer(1,3*mm),section_body]))
            elif position==0:
                divider=Table([['']],colWidths=[153*mm],rowHeights=[1.2*mm],style=TableStyle([('BACKGROUND',(0,0),(-1,-1),teal)]))
                story.append(KeepTogether([Paragraph(chapter,styles['Section']),divider,Spacer(1,3*mm),section_title,section_body]))
            else:
                story.extend([section_title,section_body])
            if idx==11:
                cells=[[Paragraph('<b>INPUT</b><br/>X-ray + 2 click maps + prior mask',styles['Caption']),Paragraph('<b>ENCODER</b><br/>24 → 48 → 96 → 192',styles['Caption']),Paragraph('<b>BOTTLENECK</b><br/>384 channels',styles['Caption']),Paragraph('<b>DECODER</b><br/>Skip connections',styles['Caption']),Paragraph('<b>OUTPUT</b><br/>Binary mask logits',styles['Caption'])]]
                t=Table(cells,colWidths=[30.6*mm]*5,rowHeights=[19*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('BOX',(0,0),(-1,-1),.6,line),('INNERGRID',(0,0),(-1,-1),.4,line),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,3*mm)]
            if idx==16:
                story.append(Paragraph('Dice = (2TP + eps) / (2TP + FP + FN + eps)  /  IoU = TP / (TP + FP + FN)<br/>Precision = TP / (TP + FP)  /  Recall = TP / (TP + FN)<br/>Training loss = BCEWithLogits + (1 - soft Dice)',styles['Caption']))
            if idx==19:
                metric_path=ROOT/'results/evaluation/metrics.json'
                if metric_path.exists():
                    m=json.loads(metric_path.read_text(encoding='utf-8'))
                    budgets=['0','1','3','5','10']
                    rows=[[Paragraph('<b>Clicks</b>',styles['Caption']),Paragraph('<b>Dice</b>',styles['Caption']),Paragraph('<b>IoU</b>',styles['Caption'])]]
                    for b in budgets:
                        q=m['interaction_budget'][b]
                        rows.append([b,f"{q['dice']:.3f}",f"{q['iou']:.3f}"])
                    t=Table(rows,colWidths=[42*mm,48*mm,48*mm],rowHeights=[7*mm]*len(rows))
                    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,line),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),('FONTSIZE',(0,0),(-1,-1),8)]))
                    story += [t,Spacer(1,3*mm)]
                    hist=ROOT/'results/graphs/training_history.png'; curve=ROOT/'results/graphs/effort_vs_dice.png'
                    figs=[]
                    for label,im in [('Training loss',hist),('Dice by click budget',curve)]:
                        if im.exists():
                            image_data=BytesIO()
                            with PILImage.open(im) as source_image:
                                source_image.convert('L').save(image_data,format='PNG')
                            image_data.seek(0)
                            figure=Table([[RLImage(image_data,width=71*mm,height=45*mm)],[Paragraph(label,styles['Caption'])]],colWidths=[76.5*mm])
                            figure.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2)]))
                            figs.append(figure)
                    if figs:
                        ft=Table([figs],colWidths=[76.5*mm]*len(figs))
                        ft.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('ALIGN',(0,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2)])); story += [ft,Spacer(1,3*mm)]
                else:
                    no_data=Table([[Paragraph('NO MEASURED RESULTS',ParagraphStyle('NodataLabel',parent=styles['Caption'],fontName='Helvetica-Bold',textColor=teal)),Paragraph('No evaluation metrics were found. This report section must be regenerated after a real evaluation run.',styles['BodyClean'])]],colWidths=[39*mm,113*mm])
                    no_data.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('BOX',(0,0),(-1,-1),.7,line),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)])); story += [no_data,Spacer(1,3*mm)]
        story.append(Spacer(1,2*mm))
    def page(canvas,doc):
        canvas.saveState()
        if doc.page>1:
            canvas.setFont('Helvetica-Bold',7.5); canvas.setFillColor(ink); canvas.drawString(21*mm,A4[1]-14*mm,'INTERACTIVE MEDICAL IMAGE SEGMENTATION')
        canvas.setStrokeColor(line); canvas.setLineWidth(.7); canvas.line(21*mm,16*mm,189*mm,16*mm)
        canvas.setFont('Helvetica',8); canvas.setFillColor(muted); canvas.drawString(21*mm,10*mm,'CAI3400 Image Analysis | 2026'); canvas.drawRightString(189*mm,10*mm,str(doc.page)); canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)

if __name__=='__main__':
    create_pdf(); print('Generated academic PDF. Build the editable presentation with build_presentation.mjs.')
