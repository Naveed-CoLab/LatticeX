import pandas as pd, numpy as np
from src.ingest.edgar import extract
def fact(start,end,val,filed,form): return {"start":start,"end":end,"val":val,"filed":filed,"form":form}
def make():
    rev=[]; ni=[]; cfo=[]
    qrev={2014:[100,110,120,130],2015:[120,130,140,150],2016:[150,160,170,180]}
    for y,qs in qrev.items():
        ends=[f"{y}-03-31",f"{y}-06-30",f"{y}-09-30",f"{y}-12-31"]; filed=[f"{y}-04-25",f"{y}-07-25",f"{y}-10-25",f"{y+1}-02-20"]
        for k in range(4):
            ytd=sum(qs[:k+1]); form="10-K" if k==3 else "10-Q"
            rev.append(fact(f"{y}-01-01",ends[k],ytd,filed[k],form)); ni.append(fact(f"{y}-01-01",ends[k],ytd*0.1,filed[k],form))
            cfo.append(fact(f"{y}-01-01",ends[k],ytd*0.15,filed[k],form))
    inst=lambda v: [{"end":f"{y}-{m}","val":v*(1+0.1*(y-2014)),"filed":f"{y}-{f}","form":"10-Q"} for y in (2014,2015,2016) for m,f in (("03-31","04-25"),("06-30","07-25"),("09-30","10-25"))]
    j={"cik":1,"facts":{"us-gaap":{"Revenues":{"units":{"USD":rev}},"NetIncomeLoss":{"units":{"USD":ni}},
       "NetCashProvidedByUsedInOperatingActivities":{"units":{"USD":cfo}},"Assets":{"units":{"USD":inst(1000)}},
       "StockholdersEquity":{"units":{"USD":inst(400)}},"Liabilities":{"units":{"USD":inst(600)}}},
       "dei":{"EntityCommonStockSharesOutstanding":{"units":{"shares":inst(10)}}}}}
    return extract(j,"FAKE")
