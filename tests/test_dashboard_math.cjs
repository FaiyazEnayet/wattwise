const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const W=require('../dashboard_ui/decisions.js');
const D=JSON.parse(fs.readFileSync(path.join(__dirname,'../demo/analysis.json'),'utf8'));
const close=(a,b,eps=1e-7)=>assert.ok(Math.abs(a-b)<eps,`${a} differs from ${b}`);
const bill=D.bills.at(-1);
test('zero intervention has zero savings and identical load',()=>{
 const r=W.simulate(D.intervals,bill,0,0,4,D.holidays);
 close(r.savings,0);close(r.totalKwhBefore,r.totalKwhAfter);
});
test('shifting conserves energy and respects receiving capacity',()=>{
 const r=W.simulate(D.intervals,bill,.4,0,4,D.holidays);
 close(r.totalKwhBefore,r.totalKwhAfter);assert.ok(r.savings>=0);
 r.rows.forEach((v,i)=>{if(r.modified[i]>v.kwh+1e-8)assert.ok(r.modified[i]<=1+1e-8);});
});
test('combined reduction and shifting do not double count energy',()=>{
 const r=W.simulate(D.intervals,bill,.5,20,4,D.holidays);
 close(r.totalKwhBefore-r.totalKwhAfter,r.removed);
 assert.ok(r.modified.every(v=>v>=-1e-8));
 const extreme=W.simulate(D.intervals,bill,1,10000,4,D.holidays);
 close(extreme.totalKwhAfter,0);close(extreme.shifted,0);
});
test('full shifting cannot exceed historical premium ceiling',()=>{
 const r=W.simulate(D.intervals,bill,1,0,100,D.holidays);
 close(r.savings,bill.shift_ceiling);
});
test('JavaScript and Python tariff replay agree',()=>{
 const rows=W.reprice(D.intervals,.139822,D.holidays);
 for(const r of rows)close(r.total,D.tariffs.find(x=>x.plan===r.plan).subtotal_usd,1e-6);
});
test('task scheduling preserves consecutive cycle and obeys deadline',()=>{
 const r=W.schedule('2026-10-20',1,1.5,15,24,.139822,D.holidays);
 assert.equal(r.best.slot,19*4);close(r.energy,1.5);
 assert.ok(r.best.slot+r.length<=96);assert.ok(r.best.cost<r.earliest.cost);
 assert.throws(()=>W.schedule('2026-10-20',1,5,15,19,.1,D.holidays));
});
test('holiday tasks do not claim an artificial peak saving',()=>{
 const r=W.schedule('2025-11-27',1,1,15,24,.139822,D.holidays);
 close(r.best.cost,r.earliest.cost);
});
test('bill attribution reconciles to total accounting change',()=>{
 const a=D.bills.at(-2),b=D.bills.at(-1),v=W.shapley(a,b);
 close(W.sum(v.values),b.reconstructed-a.reconstructed);
});
test('invalid nonfinite intervention inputs fail clearly',()=>{
 assert.throws(()=>W.simulate(D.intervals,bill,NaN,0,4,D.holidays));
 assert.throws(()=>W.simulate(D.intervals,bill,.2,-2,4,D.holidays));
 assert.throws(()=>W.reprice(D.intervals,-.1,D.holidays));
});
test('intervention matching uses earlier controls and refuses weak overlap',()=>{
 const r=W.matchIntervention(D.daily,D.weather,'2026-09-10');
 if(r.ready){assert.ok(r.matched>=14);for(const m of r.matches)assert.ok(m.controls.every(d=>d<'2026-09-10'));}
 const early=W.matchIntervention(D.daily,D.weather,'2025-09-20');
 assert.equal(early.ready,false);
});
test('perfect information is valuable only when the optimal action can change',()=>{
 const r=W.informationValue(0,200,.5,.25,50,1,0);
 close(r.expected,-25);close(r.evpi,0);
 const v=W.informationValue(0,400,.5,.25,50,1,0);
 close(v.expected,0);close(v.evpi,25);
 close(W.informationValue(400,500,.5,.25,50,1,0).evpi,0);
 assert.throws(()=>W.informationValue(500,400,.5,.25,50,1,0));
});
