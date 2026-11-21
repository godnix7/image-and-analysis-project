import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const out = path.resolve('results/presentation-preview');
await fs.mkdir(out, { recursive: true });
const evalSummary = JSON.parse(await fs.readFile('results/evaluation/metrics.json', 'utf8'));
const p = Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#14273F',teal:'#087F83',aqua:'#28B7B0',ink:'#14283B',slate:'#46596A',muted:'#667686',paper:'#F6F7F4',white:'#FFFFFF',line:'#D6DDDA',pale:'#E4EEEB',blue:'#1E4961'};
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:'none',width:0}})}
function txt(s,t,x,y,w,h,size,color,bold=false,align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
 q.text=t;q.text.style={fontSize:size,typeface:'Aptos',color,bold,alignment:align};q.text.insets=0;return q;
}
function head(s,k,title,n){s.background.fill=C.paper;rect(s,0,0,1280,8,C.teal);txt(s,k.toUpperCase(),60,30,800,24,'11pt',C.teal,true);txt(s,title,60,70,1120,54,'27pt',C.navy,true);txt(s,String(n).padStart(2,'0'),1180,36,45,22,'11pt',C.muted,true);rect(s,60,682,1160,1,C.line);txt(s,'KADEMANE NISCHAY VENKATESH',60,692,600,18,'9pt',C.muted,true)}

// 1 — cover
{
 const s=p.slides.add();s.background.fill=C.paper;rect(s,0,0,14,720,C.teal);
 txt(s,'MEDICAL IMAGING  /  INTERACTIVE SEGMENTATION',64,62,620,28,'12pt',C.teal,true);
 txt(s,'A better first mask.\nA few precise corrections.',64,150,650,150,'34pt',C.navy,true);
 txt(s,'Interactive lung-field segmentation\nwith click-guided U-Net refinement',68,338,560,76,'19pt',C.slate);
 rect(s,68,468,52,4,C.teal);txt(s,'KADEMANE NISCHAY VENKATESH',68,500,530,30,'15pt',C.navy,true);
 txt(s,'Neural Networks  ·  Deep Learning  ·  Medical Imaging',68,538,560,26,'12pt',C.muted);
 rect(s,770,0,510,720,C.navy);
 txt(s,'THE WORKFLOW',802,72,430,24,'11pt','#B5C6CE',true);
 const cards=[
  {y:126,n:'01',t:'CHEST X-RAY',d:'Grayscale image input'},
  {y:292,n:'02',t:'CLICK GUIDANCE',d:'Add missed area or remove spill'},
  {y:458,n:'03',t:'LUNG MASK',d:'Inspect and export the result'}
 ];
 for(const c of cards){
  rect(s,802,c.y,430,126,'#1E4961');
  rect(s,826,c.y+27,48,4,C.aqua);
  txt(s,c.n,826,c.y+44,54,32,'16pt','#B5C6CE',true);
  txt(s,c.t,890,c.y+36,300,30,'15pt',C.white,true);
  txt(s,c.d,890,c.y+70,300,28,'11pt','#D9E1E3');
 }
}
// 2 — problem
{
 const s=p.slides.add();head(s,'Problem','Why make segmentation interactive?',2);
 const x=[64,470,876],title=['TRACE','PREDICT','CORRECT'],big=['01','02','03'],body=[
  'Manual contours require many boundary decisions from an annotator.',
  'A one-shot model can miss edges when anatomy or contrast varies.',
  'Sparse positive and negative clicks can steer a mask toward the intended boundary.'
 ];
 for(let i=0;i<3;i++){txt(s,big[i],x[i],170,90,58,'30pt',i===2?C.teal:'#CBD6D6',true);txt(s,title[i],x[i],247,320,30,'15pt',C.navy,true);rect(s,x[i],291,300,3,i===2?C.teal:'#B8C5C7');txt(s,body[i],x[i],320,320,105,'15pt',C.slate)}
 rect(s,64,500,1152,105,C.navy);txt(s,'RESEARCH QUESTION',88,520,210,22,'10pt','#69D2CC',true);txt(s,'How does the number of correction clicks change mask quality?',88,550,1050,38,'21pt',C.white,true);
 txt(s,'Study unit: one radiograph · one binary lung-field mask · repeated model refinement',64,628,1050,24,'11pt',C.muted);
}
// 3 — architecture
{
 const s=p.slides.add();head(s,'System','One image, four channels, one refined mask',3);
 const box=(x,y,w,h,label,detail,fill)=>{rect(s,x,y,w,h,fill);txt(s,label,x+14,y+12,w-25,28,'14pt',C.white,true);txt(s,detail,x+14,y+46,w-25,100,'11pt','#EAF2F1')};
 box(60,186,235,142,'INPUT','Grayscale X-ray\nnormalized to [0, 1]',C.blue);
 box(60,350,235,184,'GUIDANCE','Positive click map\nNegative click map\nPrevious probability mask',C.teal);
 box(405,205,230,250,'ENCODER','24 → 48 → 96 → 192\n3×3 conv · BN · ReLU\nmax-pooling',C.blue);
 box(690,205,205,250,'DECODER','192 → 96 → 48 → 24\ntranspose conv\nskip connections','#255D73');
 box(1015,250,205,150,'OUTPUT','One logit map\nsigmoid → probability',C.teal);
 for(const [a,b] of [[295,392],[635,676],[895,1002]])rect(s,a,320,b-a,3,'#7F9698');
 txt(s,'Concatenate into a 4-channel tensor',60,570,520,27,'13pt',C.navy,true);
 txt(s,'Clicks are Gaussian maps; each new prediction becomes the next step’s prior mask.',60,610,1110,38,'13pt',C.slate);
}
// 4 — data/training
{
 const s=p.slides.add();head(s,'Method','From paired images to iterative training',4);
 const steps=[{x:64,n:'01',t:'PAIR',b:'Find X-rays and masks\nby normalized filename\nstem'},{x:350,n:'02',t:'PREPARE',b:'Grayscale · resize 256²\nimage: bilinear\nmask: nearest-neighbor'},{x:636,n:'03',t:'SPLIT',b:'70% train\n15% validation\n15% held-out test'},{x:922,n:'04',t:'OPTIMIZE',b:'AdamW + BCE/Dice\nReduceLROnPlateau\nearly stopping'}];
 for(const d of steps){txt(s,d.n,d.x,175,70,48,'26pt','#B9C9C8',true);txt(s,d.t,d.x,236,235,30,'14pt',C.navy,true);rect(s,d.x,275,210,3,C.teal);txt(s,d.b,d.x,301,235,88,'13pt',C.slate)}
 rect(s,64,430,1152,1,'#CAD3D1');txt(s,'TRAINING SIGNAL',64,460,320,22,'10pt',C.teal,true);txt(s,'L = BCEWithLogits + (1 − soft Dice)',64,496,650,38,'21pt',C.navy,true);
 txt(s,'After each prediction, sample the next positive or negative training click from the largest current error region.',64,551,800,58,'13pt',C.slate);
 rect(s,940,466,276,116,C.pale);txt(s,'REPRODUCIBLE',960,486,220,20,'10pt',C.teal,true);txt(s,'Seed 42\nPair-level split\nBest checkpoint: val Dice',960,516,245,58,'13pt',C.navy,true);
}
// 5 — measured held-out performance
{
 const s=p.slides.add();head(s,'Results','Held-out Dice rises sharply after one click',5);
 txt(s,`NLM MONTGOMERY  /  ${evalSummary.test_count} TEST IMAGES  /  SEED 42`,64,145,700,24,'10pt',C.teal,true);
 const budgets=[0,1,3,5,10],xs=[64,294,524,754,984];
 for(let i=0;i<budgets.length;i++){
  const b=String(budgets[i]),m=evalSummary.interaction_budget[b];
  rect(s,xs[i],190,204,170,i===1?C.navy:C.white);rect(s,xs[i],190,204,4,i===1?C.aqua:C.teal);
  txt(s,budgets[i]===1?'1 CLICK':`${budgets[i]} CLICKS`,xs[i]+14,210,174,22,'10pt',i===1?'#69D2CC':C.teal,true);
  txt(s,`Dice  ${m.dice.toFixed(3)}`,xs[i]+14,247,178,34,'20pt',i===1?C.white:C.navy,true);
  txt(s,`IoU     ${m.iou.toFixed(3)}`,xs[i]+14,292,178,26,'14pt',i===1?'#E3EEEE':C.slate,true);
 }
 rect(s,64,393,520,153,C.navy);txt(s,'INITIAL → ONE CLICK',88,415,300,22,'10pt','#69D2CC',true);
 txt(s,`${evalSummary.interaction_budget['0'].dice.toFixed(3)}  →  ${evalSummary.interaction_budget['1'].dice.toFixed(3)}`,88,449,448,44,'28pt',C.white,true);
 txt(s,'Mean Dice on the same held-out images',88,503,430,24,'12pt','#D9E3E5');
 rect(s,612,393,604,153,C.pale);txt(s,'HOW TO READ THIS',638,415,300,22,'10pt',C.teal,true);
 txt(s,'Later budgets are not guaranteed to improve: sampled oracle clicks can over-correct. This is an offline simulation, not a user study.',638,450,540,76,'14pt',C.navy,true);
 txt(s,'Each point is an average over the 21-image test split. Oracle clicks use test masks; scores are research evidence, not clinical validation.',64,584,1120,42,'11pt',C.muted);
}
// 6 — application
{
 const s=p.slides.add();head(s,'Application','A clear path from image to correction',6);
 const x=[64,350,636,922],items=[{a:'01  UPLOAD',b:'Load a chest X-ray\nValidate file and\nshow original image'},{a:'02  INSPECT',b:'Review current mask\nand overlay before\nchanging the result'},{a:'03  CORRECT',b:'Choose positive or\nnegative clicks; refine\nwith previous mask'},{a:'04  EXPORT',b:'Save mask and overlay;\nshow metrics only when\na reference mask exists'}];
 for(let i=0;i<items.length;i++){rect(s,x[i],190,250,4,i===2?C.teal:C.blue);txt(s,items[i].a,x[i],220,255,30,'13pt',C.navy,true);txt(s,items[i].b,x[i],267,245,100,'13pt',C.slate);if(i<3)rect(s,x[i]+256,201,25,2,'#AAB8B8')}
 rect(s,64,438,1152,1,'#CCD6D4');txt(s,'DESIGN CHOICES',64,468,280,22,'10pt',C.teal,true);txt(s,'A held-out example opens with its reference and live scores. Positive and negative clicks update the mask in place. Upload mode supports radiographs without benchmark labels.',64,507,1060,78,'16pt',C.navy);
 txt(s,'Streamlit interface · trained U-Net checkpoint · mask and overlay exports',64,613,1100,28,'11pt',C.muted);
}
// 7 — close
{
 const s=p.slides.add();s.background.fill=C.navy;rect(s,0,0,14,720,C.aqua);txt(s,'INTERACTIVE MEDICAL IMAGE SEGMENTATION',72,116,940,28,'12pt','#69D2CC',true);txt(s,'THANK\nYOU',72,218,720,200,'42pt',C.white,true);rect(s,72,456,78,4,C.aqua);txt(s,'Minimal input. Iterative refinement. Measured effort.',72,492,1000,44,'20pt','#D9E3E5');txt(s,'KADEMANE NISCHAY VENKATESH',72,610,900,30,'14pt',C.white,true);txt(s,'Neural Networks  ·  Deep Learning  ·  Medical Image Processing',72,648,950,24,'11pt','#B5C6CE');
}

for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(out,`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));
}
const pptx=await PresentationFile.exportPptx(p);await pptx.save('Interactive_Medical_Image_Segmentation_Presentation.pptx');
console.log(`Exported ${p.slides.items.length} slides.`);
