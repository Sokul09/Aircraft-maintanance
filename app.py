import os
import json
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AeroSense | Aircraft RUL Explorer", page_icon="✈️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aircraft_rul_random_forest.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "rul_features.pkl")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "aircraft_rul_predictions.csv")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: #07111f; color: #e8f0fa; font-family: Inter, sans-serif; }
[data-testid="stHeader"] { background: rgba(7,17,31,.9); }
.block-container { padding-top: 1.15rem; max-width: 1600px; }
section[data-testid="stSidebar"] { background: #0b1728; border-right: 1px solid #1e334c; }
.hero { padding: 1.2rem 1.35rem; border: 1px solid #1d3651; border-radius: 18px; background: linear-gradient(120deg,#10243a,#0b1728 65%,#0d2940); margin-bottom: 1rem; }
.eyebrow { color:#55d7ff; font-size: .75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { margin:.3rem 0 .25rem; font-size:2rem; color:#f2f7ff; }
.hero p { margin:0; color:#9fb2c9; }
.metric { border:1px solid #1d3651; border-radius:14px; background:#0b1a2b; padding:14px 16px; min-height:100px; }
.metric .label { font-size:.78rem; color:#8fa8c2; }
.metric .value { font-size:1.5rem; font-weight:800; margin-top:.35rem; color:#eef7ff; }
.smallnote { color:#8fa8c2; font-size:.8rem; }
div[data-testid="stNumberInput"] label { color:#b8cbe0; }
.stButton button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✈️ AeroSense")
    st.caption("AIRCRAFT HEALTH INTELLIGENCE")
    st.divider()
    st.markdown("**Explorer controls**")
    st.write("• Hover over a component to identify it")
    st.write("• Click a component to pin its details")
    st.write("• Drag to rotate · scroll to zoom")
    st.divider()
    st.markdown("**Condition legend**")
    st.markdown("🟢 **Healthy** — above 50 cycles")
    st.markdown("🟡 **Warning** — 20–50 cycles")
    st.markdown("🔴 **Critical** — below 20 cycles")
    st.divider()
    st.caption("Educational demonstration using simulated NASA C-MAPSS FD001 data. Not for real aircraft maintenance decisions.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Predictive maintenance workspace</div>
  <h1>Aircraft Digital Explorer</h1>
  <p>Explore aircraft systems in 3D and inspect engine Remaining Useful Life (RUL) predictions.</p>
</div>
""", unsafe_allow_html=True)

# Procedural Three.js aircraft: each modeled part has its own mesh and hover/click metadata.
aircraft_html = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box} body{margin:0;background:#081422;color:#eaf3ff;font-family:Inter,Arial,sans-serif}
.shell{display:grid;grid-template-columns:minmax(0,1fr) 300px;min-height:620px;border:1px solid #20364d;border-radius:16px;overflow:hidden;background:#091727}
.stage{position:relative;min-height:620px;background:radial-gradient(ellipse at 50% 42%,#17314b 0%,#0b1a2b 50%,#081422 100%);overflow:hidden}
#view{position:absolute;inset:0} .hud{position:absolute;z-index:3;left:16px;top:15px;pointer-events:none}
.kicker{font-size:10px;letter-spacing:2px;color:#58d8ff;font-weight:800}.hud h2{font-size:17px;margin:5px 0;color:#eff7ff}.hud p{font-size:11px;color:#8da8c4;margin:0}
.controls{position:absolute;z-index:4;right:14px;top:14px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
button{font:inherit;color:#cfe7fb;background:#10263b;border:1px solid #2a4b67;border-radius:8px;padding:8px 10px;font-size:11px;cursor:pointer}
button:hover,button.active{background:#124666;border-color:#43cfff;color:white}
#tip{display:none;position:absolute;z-index:8;pointer-events:none;padding:9px 11px;background:#07111fee;border:1px solid #42cfff;border-radius:9px;box-shadow:0 8px 30px #0008;max-width:220px}
#tip b{display:block;font-size:12px;color:#5ddcff;margin-bottom:3px}#tip span{font-size:11px;color:#c0d0e2;line-height:1.4}
.badge{position:absolute;bottom:15px;left:16px;z-index:3;border:1px solid #25445e;border-radius:9px;background:#07111fbd;padding:8px 10px;font-size:10px;color:#91abc5}
.panel{padding:19px 17px;border-left:1px solid #20364d;background:#0b1929;overflow:auto}
.paneltop{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}.paneltop h3{font-size:13px;letter-spacing:1px;margin:0;color:#8ca8c4;text-transform:uppercase}
.status{font-size:10px;color:#54d9ff;border:1px solid #245b77;border-radius:99px;padding:5px 8px}
#partname{font-size:22px;font-weight:800;line-height:1.15;margin:6px 0 8px;color:#f1f7ff}
#category{color:#5ddcff;font-size:10px;letter-spacing:1px;text-transform:uppercase;font-weight:700}
#desc{font-size:12px;color:#afc1d4;line-height:1.65;margin:10px 0 17px}
.section{border-top:1px solid #1e344a;padding:13px 0}.section h4{font-size:10px;color:#7f9bb7;letter-spacing:1.3px;text-transform:uppercase;margin:0 0 8px}
.section p{font-size:11px;color:#d2dfec;line-height:1.55;margin:0}
.chip{display:inline-block;background:#102b41;color:#7fddff;border:1px solid #234d69;border-radius:6px;padding:5px 7px;margin:3px 3px 0 0;font-size:10px}
.search{width:100%;padding:9px 10px;background:#071422;border:1px solid #25415a;border-radius:8px;color:#eaf3ff;outline:none;margin:9px 0 8px;font-size:11px}
.list{display:flex;flex-wrap:wrap;gap:5px;max-height:122px;overflow:auto}
.partbtn{padding:6px 8px;font-size:10px;white-space:nowrap}
.footer{color:#6985a2;font-size:10px;margin-top:12px;line-height:1.5}
@media(max-width:800px){.shell{grid-template-columns:1fr}.stage{min-height:470px}.panel{border-left:0;border-top:1px solid #20364d}.controls{max-width:70%}}
</style></head><body>
<div class="shell">
 <div class="stage" id="stage">
  <div id="view"></div>
  <div class="hud"><div class="kicker">AEROSENSE / MODEL VIEW</div><h2>Interactive aircraft systems</h2><p>Hover to identify · click to inspect</p></div>
  <div class="controls">
   <button id="iso" class="active">Isometric</button><button id="top">Top</button><button id="side">Side</button><button id="front">Front</button><button id="reset">Reset view</button>
  </div>
  <div id="tip"><b id="tipname"></b><span id="tipdesc"></span></div>
  <div class="badge">PROCEDURAL DEMO MODEL · component meshes are selectable</div>
 </div>
 <aside class="panel">
  <div class="paneltop"><h3>Component inspector</h3><span class="status">INTERACTIVE</span></div>
  <input id="search" class="search" placeholder="Search components..." />
  <div id="partlist" class="list"></div>
  <div class="section" style="margin-top:12px;border-top:1px solid #294158;padding-top:16px">
   <div id="category">AIRFRAME</div><div id="partname">Fuselage</div>
   <p id="desc">The main body of the aircraft that houses the cockpit, passengers, and systems.</p>
  </div>
  <div class="section"><h4>Primary function</h4><p id="function">Provides the main structure and connects major aircraft sections.</p></div>
  <div class="section"><h4>Maintenance considerations</h4><p id="maintenance">Inspect for structural damage, corrosion, fastener condition, and signs of fatigue according to approved maintenance documentation.</p></div>
  <div class="section"><h4>Relevant monitoring</h4><div id="sensors"><span class="chip">Visual inspection</span><span class="chip">Structural inspection</span></div></div>
  <div class="section"><h4>Predictive status</h4><p id="prediction">No component-specific prediction is connected in this explorer. Engine RUL appears in the model results section below when the trained model is available.</p></div>
  <div class="footer">The 3D geometry is a procedural demonstration, not a certified aircraft model. Use approved technical documentation for real maintenance.</div>
 </aside>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
const data = {
 fuselage:{cat:'Airframe',desc:'The main body of the aircraft that houses the cockpit, passengers, cargo, and systems.',fn:'Provides the central structure and connects the wings, tail, and landing gear.',maint:'Inspect for structural damage, corrosion, fastener condition, and signs of fatigue according to approved maintenance documentation.',s:['Visual inspection','Structural inspection','Corrosion checks']},
 cockpit:{cat:'Flight deck',desc:'The flight deck contains the controls and displays used by the flight crew.',fn:'Allows pilots to control and monitor aircraft systems.',maint:'Check windows, seals, controls, displays, and flight-deck systems using approved procedures.',s:['Display diagnostics','Window inspection']},
 radome:{cat:'Nose / avionics',desc:'The nose radome protects the weather radar antenna while allowing radar signals to pass through.',fn:'Protects radar equipment from weather and aerodynamic loads.',maint:'Check for cracks, erosion, moisture ingress, and impact damage.',s:['Visual inspection','Radar system checks']},
 wingL:{cat:'Wing structure',desc:'The left wing generates lift and houses fuel and control surfaces.',fn:'Generates lift and supports fuel tanks and flight-control surfaces.',maint:'Inspect the skin, joints, control surfaces, and visible signs of damage or leakage.',s:['Visual inspection','Fuel leak checks','Structural checks']},
 wingR:{cat:'Wing structure',desc:'The right wing generates lift and houses fuel and control surfaces.',fn:'Generates lift and supports fuel tanks and flight-control surfaces.',maint:'Inspect the skin, joints, control surfaces, and visible signs of damage or leakage.',s:['Visual inspection','Fuel leak checks','Structural checks']},
 flapL:{cat:'High-lift system',desc:'The left flap extends to increase lift and drag during takeoff and landing.',fn:'Changes wing camber to support lower-speed flight.',maint:'Inspect hinges, tracks, actuators, and movement under approved maintenance procedures.',s:['Actuator checks','Position indication']},
 flapR:{cat:'High-lift system',desc:'The right flap extends to increase lift and drag during takeoff and landing.',fn:'Changes wing camber to support lower-speed flight.',maint:'Inspect hinges, tracks, actuators, and movement under approved maintenance procedures.',s:['Actuator checks','Position indication']},
 aileronL:{cat:'Flight controls',desc:'The left aileron helps control aircraft roll.',fn:'Moves with the opposite aileron to control roll around the aircraft longitudinal axis.',maint:'Inspect hinges, linkages, actuator operation, and control-surface condition.',s:['Control checks','Actuator checks']},
 aileronR:{cat:'Flight controls',desc:'The right aileron helps control aircraft roll.',fn:'Moves with the opposite aileron to control roll around the aircraft longitudinal axis.',maint:'Inspect hinges, linkages, actuator operation, and control-surface condition.',s:['Control checks','Actuator checks']},
 slatL:{cat:'High-lift system',desc:'Leading-edge slats help maintain lift at higher angles of attack.',fn:'Improves low-speed aerodynamic performance.',maint:'Inspect tracks, seals, actuators, and movement.',s:['Actuator checks','Visual inspection']},
 slatR:{cat:'High-lift system',desc:'Leading-edge slats help maintain lift at higher angles of attack.',fn:'Improves low-speed aerodynamic performance.',maint:'Inspect tracks, seals, actuators, and movement.',s:['Actuator checks','Visual inspection']},
 wingletL:{cat:'Wingtip',desc:'A winglet reduces some wingtip-vortex-related aerodynamic drag.',fn:'Improves aerodynamic efficiency for the aircraft design.',maint:'Inspect attachments and surfaces for impact damage, cracks, and looseness.',s:['Visual inspection','Structural checks']},
 wingletR:{cat:'Wingtip',desc:'A winglet reduces some wingtip-vortex-related aerodynamic drag.',fn:'Improves aerodynamic efficiency for the aircraft design.',maint:'Inspect attachments and surfaces for impact damage, cracks, and looseness.',s:['Visual inspection','Structural checks']},
 engineL:{cat:'Propulsion',desc:'The left turbofan provides thrust. The project's RUL model applies to engine data, not to every aircraft part.',fn:'Produces thrust through fan, compressor, combustion, turbine, and exhaust stages.',maint:'Use engine-specific approved procedures and validated sensor data; inspect for abnormal vibration, leaks, damage, and performance changes.',s:['Temperature (EGT)','Pressure ratios','Rotor speed','Vibration']},
 engineR:{cat:'Propulsion',desc:'The right turbofan provides thrust. The project's RUL model applies to engine data, not to every aircraft part.',fn:'Produces thrust through fan, compressor, combustion, turbine, and exhaust stages.',maint:'Use engine-specific approved procedures and validated sensor data; inspect for abnormal vibration, leaks, damage, and performance changes.',s:['Temperature (EGT)','Pressure ratios','Rotor speed','Vibration']},
 fanL:{cat:'Engine internals',desc:'The front fan accelerates a large airflow and feeds the engine core and bypass stream.',fn:'Moves air through the turbofan and contributes substantially to thrust.',maint:'Inspect blades and fan case using approved engine inspection procedures.',s:['Fan speed','Vibration','Visual / borescope']},
 fanR:{cat:'Engine internals',desc:'The front fan accelerates a large airflow and feeds the engine core and bypass stream.',fn:'Moves air through the turbofan and contributes substantially to thrust.',maint:'Inspect blades and fan case using approved engine inspection procedures.',s:['Fan speed','Vibration','Visual / borescope']},
 gearN:{cat:'Landing gear',desc:'The nose landing gear supports the forward aircraft during ground operations.',fn:'Supports taxi, takeoff, landing, and steering functions.',maint:'Inspect tyres, strut, steering linkages, and signs of fluid leaks.',s:['Tyre condition','Strut inspection']},
 gearL:{cat:'Landing gear',desc:'The left main landing gear carries ground loads during taxi, takeoff, and landing.',fn:'Absorbs landing loads and supports the aircraft on the ground.',maint:'Inspect tyres, brakes, strut, wheels, and hydraulic lines.',s:['Brake wear','Tyre condition','Leak inspection']},
 gearR:{cat:'Landing gear',desc:'The right main landing gear carries ground loads during taxi, takeoff, and landing.',fn:'Absorbs landing loads and supports the aircraft on the ground.',maint:'Inspect tyres, brakes, strut, wheels, and hydraulic lines.',s:['Brake wear','Tyre condition','Leak inspection']},
 tailH:{cat:'Tail assembly',desc:'The horizontal stabilizer helps provide longitudinal stability.',fn:'Contributes to pitch stability and supports the elevator.',maint:'Inspect skin, attachment points, and control-surface hinges.',s:['Visual inspection','Control checks']},
 tailV:{cat:'Tail assembly',desc:'The vertical stabilizer helps provide directional stability.',fn:'Contributes to yaw stability and supports the rudder.',maint:'Inspect structure, attachment points, and rudder hinges.',s:['Visual inspection','Control checks']},
 rudder:{cat:'Flight controls',desc:'The rudder is a movable surface on the vertical tail.',fn:'Controls yaw around the aircraft vertical axis.',maint:'Inspect hinges, linkages, actuator, and surface condition.',s:['Actuator checks','Control checks']},
 elevator:{cat:'Flight controls',desc:'The elevator is a movable surface on the horizontal tail.',fn:'Controls pitch around the aircraft lateral axis.',maint:'Inspect hinges, linkages, actuator, and surface condition.',s:['Actuator checks','Control checks']},
 spoilerL:{cat:'Flight controls',desc:'Spoilers on the wing can reduce lift and increase drag when deployed.',fn:'Assist descent, roll control, and lift dumping after landing.',maint:'Inspect panels, hinges, actuators, and deployment indications.',s:['Actuator checks','Position indication']},
 spoilerR:{cat:'Flight controls',desc:'Spoilers on the wing can reduce lift and increase drag when deployed.',fn:'Assist descent, roll control, and lift dumping after landing.',maint:'Inspect panels, hinges, actuators, and deployment indications.',s:['Actuator checks','Position indication']},
 door:{cat:'Cabin / access',desc:'Passenger doors provide cabin access and evacuation routes.',fn:'Provides controlled entry and exit from the cabin.',maint:'Inspect seals, hinges, latches, indicators, and surrounding structure.',s:['Door indication','Seal inspection']},
 windows:{cat:'Cabin / transparencies',desc:'Cabin windows provide passenger visibility and are part of the pressure boundary.',fn:'Provides external visibility while maintaining cabin pressure integrity.',maint:'Inspect for cracks, crazing, seal condition, and damage.',s:['Visual inspection','Seal inspection']},
 antenna:{cat:'Avionics',desc:'External antennas and sensors support communication, navigation, and aircraft monitoring.',fn:'Receives or transmits signals and supplies measurements to aircraft systems.',maint:'Check attachment, condition, cabling, and system test results.',s:['System diagnostics','Visual inspection']},
 exhaustL:{cat:'Engine internals',desc:'The exhaust section directs hot gases rearward to contribute to thrust.',fn:'Discharges the engine flow after the turbine.',maint:'Inspect for cracks, distortion, hot spots, and damage under approved procedures.',s:['Exhaust temperature','Borescope / visual']},
 exhaustR:{cat:'Engine internals',desc:'The exhaust section directs hot gases rearward to contribute to thrust.',fn:'Discharges the engine flow after the turbine.',maint:'Inspect for cracks, distortion, hot spots, and damage under approved procedures.',s:['Exhaust temperature','Borescope / visual']}
};
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b1a2b);
const camera = new THREE.PerspectiveCamera(42,1,0.1,1000); camera.position.set(13,10,15);
const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(800,600); renderer.shadowMap.enabled=true; renderer.outputColorSpace=THREE.SRGBColorSpace; document.getElementById('view').appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.target.set(0,0.2,0); controls.minDistance=7; controls.maxDistance=35;
scene.add(new THREE.HemisphereLight(0xb9e6ff,0x203149,2.1));
const key = new THREE.DirectionalLight(0xffffff,2.5); key.position.set(8,15,10); scene.add(key);
const rim = new THREE.PointLight(0x22baff,45,50); rim.position.set(-8,5,-5); scene.add(rim);
const mat=(c,metal=0.45,rough=0.35)=>new THREE.MeshStandardMaterial({color:c,metalness:metal,roughness:rough});
const bodyMat=mat(0xdbe7f3,.65,.28), wingMat=mat(0xbacbdc,.55,.34), blueMat=mat(0x287ba5,.5,.32), darkMat=mat(0x182b3b,.55,.3), glassMat=mat(0x153f59,.65,.16), engineMat=mat(0x536b7d,.7,.28), accentMat=mat(0x48d8ff,.45,.28);
const objects=[]; const named={};
function addMesh(id,label,geometry,material,pos,rot){const mesh=new THREE.Mesh(geometry,material);mesh.position.set(...pos);if(rot)mesh.rotation.set(...rot);mesh.castShadow=true;mesh.receiveShadow=true;mesh.userData={id,label};scene.add(mesh);objects.push(mesh);named[id]=mesh;return mesh;}
function box(id,label,dim,pos,material=wingMat,rot){return addMesh(id,label,new THREE.BoxGeometry(...dim),material,pos,rot)}
function cyl(id,label,r1,r2,len,pos,material=engineMat,axis='x'){let g=new THREE.CylinderGeometry(r1,r2,len,24,1);if(axis==='x')g.rotateZ(Math.PI/2);if(axis==='z')g.rotateX(Math.PI/2);return addMesh(id,label,g,material,pos)}
function cone(id,label,r1,r2,len,pos,material=bodyMat,axis='x'){let g=new THREE.CylinderGeometry(r1,r2,len,32);if(axis==='x')g.rotateZ(Math.PI/2);return addMesh(id,label,g,material,pos)}
// Aircraft longitudinal axis is X; nose points toward negative X.
addMesh('fuselage','Fuselage',new THREE.CylinderGeometry(.66,.78,10.2,32,1),bodyMat,[0,0,0],[0,0,Math.PI/2]);
cone('radome','Nose cone / radome',.02,.66,1.6,[-5.85,0,0],bodyMat);
cone('cockpit','Cockpit',.24,.42,.8,[-4.85,.47,0],glassMat);
box('wingL','Left wing',[5.8,.18,2.3],[-.15,-.03,-2.3],wingMat,[0,0,-.02]);
box('wingR','Right wing',[5.8,.18,2.3],[-.15,-.03,2.3],wingMat,[0,0,.02]);
box('flapL','Left wing flap',[2.0,.12,.45],[.55,-.16,-3.25],blueMat);
box('flapR','Right wing flap',[2.0,.12,.45],[.55,-.16,3.25],blueMat);
box('aileronL','Left aileron',[1.5,.11,.36],[-1.7,-.15,-3.25],accentMat);
box('aileronR','Right aileron',[1.5,.11,.36],[-1.7,-.15,3.25],accentMat);
box('slatL','Left leading-edge slat',[2.1,.1,.28],[-1.9,.12,-3.15],bodyMat);
box('slatR','Right leading-edge slat',[2.1,.1,.28],[-1.9,.12,3.15],bodyMat);
box('wingletL','Left winglet',[.16,1.0,.52],[-.15,.42,-3.55],blueMat,[.12,0,0]);
box('wingletR','Right winglet',[.16,1.0,.52],[-.15,.42,3.55],blueMat,[.12,0,0]);
box('tailH','Horizontal stabilizer',[2.4,.13,.9],[4.35,.15,0],wingMat);
box('tailV','Vertical stabilizer',[1.7,1.45,.16],[4.25,.82,0],blueMat,[0,0,-.15]);
box('elevator','Elevators',[.8,.1,1.0],[5.1,.11,0],accentMat);
box('rudder','Rudder',[.8,1.0,.1],[4.95,1.0,0],accentMat,[0,0,-.12]);
box('spoilerL','Left spoiler',[1.05,.09,.45],[-.6,.11,-2.65],accentMat);
box('spoilerR','Right spoiler',[1.05,.09,.45],[-.6,.11,2.65],accentMat);
// Twin turbofans, placed forward and under each wing.
for(const [side,z,id] of [['Left',-1.55,'L'],['Right',1.55,'R']]){
  cyl('engine'+id,side+' engine nacelle',.65,.72,2.05,[-.55,-.72,z],engineMat);
  cyl('fan'+id,side+' engine fan / compressor',.48,.48,.12,[-1.62,-.72,z],darkMat);
  cyl('exhaust'+id,side+' engine exhaust',.38,.42,.28,[.62,-.72,z],darkMat);
  for(let i=0;i<8;i++){let a=i*Math.PI/4;box('fanBlade'+id+i,side+' fan blade',[.42,.055,.055],[-1.62,-.72+Math.cos(a)*.24,z+Math.sin(a)*.24],accentMat,[a,0,0]);}
}
// Landing gear and wheels
cyl('gearN','Nose landing gear strut',.07,.07,.9,[-3.6,-.9,0],darkMat,'y');
for(const [z,id,side] of [[-1.2,'L','Left'],[1.2,'R','Right']]){cyl('gear'+id,side+' main landing gear strut',.09,.09,1.0,[.9,-.95,z],darkMat,'y');cyl('wheel'+id,side+' landing gear wheel',.29,.29,.18,[.9,-1.45,z],darkMat,'z');}
cyl('wheelN','Nose landing gear wheel',.22,.22,.15,[-3.6,-1.4,0],darkMat,'z');
box('door','Passenger door',[.08,1.15,.5],[-2.3,.08,.68],blueMat);
for(let i=0;i<7;i++){let x=-3.7+i*.85;box('window'+i,'Cabin window',[.06,.16,.27],[x,.32,.665],glassMat);}
box('antenna','Antenna / external sensor',[.18,.28,.18],[1.1,.78,0],accentMat);
const grid=new THREE.GridHelper(30,30,0x23435d,0x152b40);grid.position.y=-1.75;scene.add(grid);
const floor=new THREE.Mesh(new THREE.CircleGeometry(12,64),new THREE.MeshBasicMaterial({color:0x091827,transparent:true,opacity:.45}));floor.rotation.x=-Math.PI/2;floor.position.y=-1.78;scene.add(floor);
let hovered=null,selected='fuselage';const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();const tip=document.getElementById('tip'),stage=document.getElementById('stage');
function restore(mesh){if(mesh&&mesh.material&&mesh.userData.originalColor!==undefined){mesh.material.emissive.setHex(0x000000);mesh.material.color.setHex(mesh.userData.originalColor);mesh.material.needsUpdate=true;}}
function highlight(mesh,color=0x39cfff){if(!mesh||!mesh.material)return;if(mesh.userData.originalColor===undefined)mesh.userData.originalColor=mesh.material.color.getHex();mesh.material.emissive.setHex(color);mesh.material.emissiveIntensity=.48;}
function showPart(id){selected=id;const d=data[id]||{cat:'Aircraft component',desc:'A modeled aircraft component for visual exploration.',fn:'Part of the aircraft system.',maint:'Inspect using approved maintenance documentation.',s:['Visual inspection']};document.getElementById('category').textContent=d.cat;document.getElementById('partname').textContent=(named[id]?.userData.label)||id;document.getElementById('desc').textContent=d.desc;document.getElementById('function').textContent=d.fn;document.getElementById('maintenance').textContent=d.maint;document.getElementById('sensors').innerHTML=d.s.map(s=>'<span class="chip">'+s+'</span>').join('');document.getElementById('prediction').textContent=(id==='engineL'||id==='engineR'||id==='fanL'||id==='fanR'||id==='exhaustL'||id==='exhaustR')?'Engine RUL is shown below only if the trained project model and valid input features are available. The NASA C-MAPSS dataset is simulated.':'No validated part-specific RUL model is connected. This component has descriptive inspection guidance only.';document.querySelectorAll('.partbtn').forEach(b=>b.classList.toggle('active',b.dataset.id===id));}
function updatePointer(e){const r=renderer.domElement.getBoundingClientRect();mouse.x=((e.clientX-r.left)/r.width)*2-1;mouse.y=-((e.clientY-r.top)/r.height)*2+1;raycaster.setFromCamera(mouse,camera);const hits=raycaster.intersectObjects(objects,false);const hit=hits.find(h=>h.object.userData.id);if(hovered&&(!hit||hit.object!==hovered)){restore(hovered);hovered=null;}if(hit){hovered=hit.object;highlight(hovered);const id=hovered.userData.id,d=data[id]||data.fuselage;document.getElementById('tipname').textContent=hovered.userData.label;document.getElementById('tipdesc').textContent=d.desc;tip.style.display='block';const sr=stage.getBoundingClientRect();tip.style.left=Math.min(e.clientX-sr.left+14,sr.width-235)+'px';tip.style.top=Math.min(e.clientY-sr.top+12,sr.height-85)+'px';renderer.domElement.style.cursor='pointer';}else{tip.style.display='none';renderer.domElement.style.cursor='grab';}}
renderer.domElement.addEventListener('pointermove',updatePointer);
renderer.domElement.addEventListener('pointerleave',()=>{if(hovered){restore(hovered);hovered=null;}tip.style.display='none';});
renderer.domElement.addEventListener('click',e=>{const r=renderer.domElement.getBoundingClientRect();mouse.x=((e.clientX-r.left)/r.width)*2-1;mouse.y=-((e.clientY-r.top)/r.height)*2+1;raycaster.setFromCamera(mouse,camera);const hit=raycaster.intersectObjects(objects,false).find(h=>h.object.userData.id);if(hit)showPart(hit.object.userData.id);});
const allIds=Object.keys(data);const list=document.getElementById('partlist');function renderList(q=''){list.innerHTML='';allIds.filter(id=>((named[id]?.userData.label)||id).toLowerCase().includes(q.toLowerCase())).forEach(id=>{let b=document.createElement('button');b.className='partbtn';b.dataset.id=id;b.textContent=named[id]?.userData.label||id;b.onclick=()=>{showPart(id);const m=named[id];if(m){controls.target.copy(m.position);}};list.appendChild(b);});}renderList();document.getElementById('search').addEventListener('input',e=>renderList(e.target.value));
function setView(v){if(v==='top')camera.position.set(0,22,.01);else if(v==='side')camera.position.set(0,3,23);else if(v==='front')camera.position.set(-24,4,0);else camera.position.set(13,10,15);controls.target.set(0,0,0);controls.update();document.querySelectorAll('.controls button').forEach(b=>b.classList.toggle('active',b.id===v));}
['iso','top','side','front'].forEach(id=>document.getElementById(id).onclick=()=>setView(id));document.getElementById('reset').onclick=()=>setView('iso');
function resize(){const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}new ResizeObserver(resize).observe(stage);resize();showPart('fuselage');
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}animate();
</script></body></html>
"""

components.html(aircraft_html, height=640, scrolling=False)

st.markdown("### Engine Remaining Useful Life")
st.markdown('<div class="smallnote">Predictions are engine-level only. Component descriptions above do not imply component-specific health predictions.</div>', unsafe_allow_html=True)

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        st.caption("Enter a valid row of engine operating and sensor measurements. Avoid default/placeholder values when interpreting results.")
        with st.form("rul_form"):
            values = {}
            cols = st.columns(3)
            for i, feature in enumerate(features):
                default = 100.0 if feature == "cycle" else 0.0
                with cols[i % 3]:
                    values[feature] = st.number_input(str(feature), value=float(default), format="%.4f", key=f"sensor_{feature}")
            submitted = st.form_submit_button("Predict engine RUL", type="primary", use_container_width=True)
        if submitted:
            row = pd.DataFrame([[values[f] for f in features]], columns=features)
            predicted_rul = max(0.0, float(model.predict(row)[0]))
            if predicted_rul > 50:
                condition, risk, recommendation = "Healthy", "Low", "Continue routine monitoring and scheduled maintenance."
            elif predicted_rul >= 20:
                condition, risk, recommendation = "Warning", "Medium", "Increase monitoring frequency and schedule preventive maintenance."
            else:
                condition, risk, recommendation = "Critical", "High", "Prioritize inspection and maintenance planning."
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric"><div class="label">PREDICTED ENGINE RUL</div><div class="value">{predicted_rul:.1f} cycles</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric"><div class="label">ENGINE CONDITION</div><div class="value">{condition}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric"><div class="label">RISK LEVEL</div><div class="value">{risk}</div></div>', unsafe_allow_html=True)
            if condition == "Healthy": st.success(recommendation)
            elif condition == "Warning": st.warning(recommendation)
            else: st.error(recommendation)
            st.caption("Thresholds are project-defined decision rules, not official NASA condition labels.")
    except Exception as exc:
        st.error(f"The trained model could not be loaded or used: {exc}")
        st.info("Check that the model file and feature list match this project environment.")
else:
    st.warning("The trained model file `aircraft_rul_random_forest.pkl` is not included in the uploaded project archive, so live RUL prediction is unavailable until you add it.")
    st.info("The interactive aircraft explorer works independently. Add the trained model file beside `app.py` to enable predictions; `rul_features.pkl` alone is not a prediction model.")
    if os.path.exists(PREDICTIONS_PATH):
        with st.expander("View existing sample prediction records"):
            sample = pd.read_csv(PREDICTIONS_PATH)
            st.caption("These are saved sample results, not live predictions from the missing model.")
            st.dataframe(sample.head(20), use_container_width=True)

st.divider()
st.caption("AeroSense Aircraft Explorer · Three.js procedural model · Streamlit · Educational use only. Not approved for operational aircraft maintenance.")
