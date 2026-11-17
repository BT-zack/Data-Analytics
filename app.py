import math
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
BASE=Path(__file__).resolve().parent
bundle=joblib.load(BASE/"365_real_estate_xgb_model.joblib"); model=bundle["pipeline"]; meta=bundle["metadata"]; tx=pd.read_csv(BASE/"clean_transactions.csv",parse_dates=["instance_date"])
st.set_page_config(page_title="365 Real Estate Valuation",page_icon="🏙️",layout="wide")
st.title("365 Real Estate — Dubai Property Valuation"); st.caption("Residential DLD sales model | 2020–2025")
with st.sidebar:
 st.header("Property details")
 community=st.selectbox("Community",meta["communities"],index=meta["communities"].index("Jumeirah Village Circle"))
 property_type=st.selectbox("Property type",meta["property_types"],index=meta["property_types"].index("Flat"))
 rooms=st.selectbox("Bedrooms / rooms",meta["rooms"],index=meta["rooms"].index("2 B/R"))
 size_sqft=st.number_input("Size (sq ft)",min_value=100.0,max_value=500000.0,value=1000.0,step=25.0)
 off_plan=st.checkbox("Off-plan property",value=False); parking=st.checkbox("Has parking",value=True)
 market_year=st.selectbox("Market year used by model",list(range(2020,2026)),index=5); market_month=st.selectbox("Market month used by model",list(range(1,13)),index=8)
 run=st.button("Estimate price",type="primary",use_container_width=True)
if run:
 row=pd.DataFrame([{"size_sqft":size_sqft,"property_sub_type_en":property_type,"rooms_en":rooms,"community":community,"reg_type_en":"Off-Plan Properties" if off_plan else "Existing Properties","has_parking":1 if parking else 0,"year":market_year,"month_sin":math.sin(2*math.pi*market_month/12),"month_cos":math.cos(2*math.pi*market_month/12)}])
 estimate=math.exp(float(model.predict(row)[0])); low=estimate*meta["range_multipliers"][0]; high=estimate*meta["range_multipliers"][1]
 c1,c2,c3=st.columns(3); c1.metric("Estimated sale price",f"AED {estimate:,.0f}"); c2.metric("Lower range",f"AED {low:,.0f}"); c3.metric("Upper range",f"AED {high:,.0f}")
 st.info("Range = empirical 90% model-error range from the held-out test set; it is not a formal confidence interval.")
 st.subheader("Recent comparable transactions")
 comps=tx[tx.community.eq(community)].copy(); focused=comps[comps.property_sub_type_en.eq(property_type)&comps.rooms_en.eq(rooms)].copy()
 if len(focused)>=3: comps=focused
 comps=comps.sort_values("instance_date",ascending=False).head(5); comps["Sale price (AED)"]=comps.actual_worth.round().astype(int); comps["Size (sq ft)"]=comps.size_sqft.round().astype(int); comps["Price / sq ft (AED)"]=(comps.actual_worth/comps.size_sqft).round().astype(int)
 d=comps[["instance_date","property_sub_type_en","rooms_en","Size (sq ft)","Sale price (AED)","Price / sq ft (AED)","reg_type_en"]].copy(); d.columns=["Date","Type","Rooms","Size (sq ft)","Sale price (AED)","Price / sq ft (AED)","Registration"]; st.dataframe(d,use_container_width=True,hide_index=True)
st.divider(); st.caption("Scope: residential DLD sales for Flat, Villa and Hotel Apartment records in 22 standardised communities. Target: log(actual_worth).")
