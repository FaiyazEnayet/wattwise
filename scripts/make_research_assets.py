"""Render README charts and a growing observed-energy animation."""
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'demo/analysis.json').read_text(encoding='utf-8'))
BG='#0b1915';TEXT='#f1ecd9';GREEN='#9ac7aa';AMBER='#e4b66e';COPPER='#c68e73'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'text.color':TEXT,
 'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT,'axes.edgecolor':'#355246',
 'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':BG})
fig,axs=plt.subplots(2,2,figsize=(13,7.8),layout='constrained')
fig.suptitle('WattWise  |  Measurements and held-out evidence',fontsize=18,x=.04,ha='left')
daily=D['daily'];dates=np.array([np.datetime64(r['date']) for r in daily]);energy=np.array([r['kwh'] for r in daily])
axs[0,0].plot(dates,energy,color=GREEN,lw=1)
axs[0,0].set(title='Original daily meter energy',ylabel='kWh / recorded day')
tariffs=D['tariffs'];axs[0,1].bar(['180','194','195'],[r['subtotal_usd'] for r in tariffs],color=[GREEN,COPPER,AMBER])
axs[0,1].set(title='Standardized annual tariff replay',ylabel='USD energy + service subtotal',xlabel='Rate plan')
axs[0,1].text(.02,.98,'2026 delivery; fixed $0.139822/kWh base supply\nTaxes and adjustments excluded',transform=axs[0,1].transAxes,va='top',fontsize=8)
f=D['forecast'];axs[1,0].barh(['Weekly persistence','Weekday median','Ridge autoregression'],[m['holdout_mae'] for m in f['models']],color=[GREEN,COPPER,AMBER])
axs[1,0].set(title='Untouched final 28-day forecast holdout',xlabel='MAE  |  kWh/day; lower is better')
for i,v in enumerate(D['shapes']['centers']):axs[1,1].plot(range(24),v,color=[GREEN,AMBER,COPPER][i],label=f'Pattern {i+1}: {D["shapes"]["counts"][i]} days')
axs[1,1].set(title='Descriptive daily load-shape groups',ylabel='Share of daily kWh (%)',xlabel='Local hour')
axs[1,1].legend(fontsize=8,facecolor=BG,labelcolor=TEXT,frameon=False)
for ax in axs.flat:ax.grid(axis='y',color='#355246',alpha=.4);ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'docs/analysis-evidence.png',dpi=140);plt.close(fig)
frames=[]
for stop in np.linspace(12,len(daily),32).astype(int):
    fig,ax=plt.subplots(figsize=(10,4.9),layout='constrained')
    ax.plot(dates[:stop],energy[:stop],color=AMBER,lw=1.5)
    ax.set(xlim=(dates[0],dates[-1]),ylim=(0,34),ylabel='Original daily energy (kWh)')
    ax.set_title('WattWise  |  A year of measured electricity',loc='left',pad=20,fontsize=18)
    ax.text(0,.97,'37,976 original intervals  |  '+str(dates[stop-1]),transform=ax.transAxes,va='top',fontsize=10,color=GREEN)
    ax.text(0,-.2,'Historical daily totals revealed chronologically; no synthetic household readings.',transform=ax.transAxes,fontsize=8)
    ax.grid(axis='y',color='#355246',alpha=.4);ax.spines[['top','right']].set_visible(False)
    buffer=io.BytesIO();fig.savefig(buffer,dpi=90);plt.close(fig);buffer.seek(0)
    frames.append(Image.open(buffer).convert('RGB').copy())
frames[0].save(ROOT/'docs/analysis-demo.gif',save_all=True,append_images=frames[1:],duration=[100]*31+[1800],loop=0,optimize=True)
print('Rendered source-backed research figure and meter-history animation')
