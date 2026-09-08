import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "uber_fare_model.pkl"
META_PATH = BASE / "models" / "model_metadata.json"
DATA_PATH = BASE / "dataset" / "uber_fare_cleaned.csv"

st.set_page_config(page_title="Uber Fare Intelligence", page_icon="🚕", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.hero {padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(135deg,#111827,#1f2937);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:2.1rem}.hero p{margin:.35rem 0 0;color:#d1d5db}
.small-note {font-size:.86rem;color:#6b7280}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load(MODEL_PATH)
@st.cache_data
def load_data():
    d=pd.read_csv(DATA_PATH)
    d['pickup_time']=pd.to_datetime(d['pickup_time'],errors='coerce')
    d['drop_time']=pd.to_datetime(d['drop_time'],errors='coerce')
    return d
@st.cache_data
def load_meta(): return json.loads(META_PATH.read_text(encoding='utf-8'))

model=load_model(); df=load_data(); meta=load_meta()

st.markdown('<div class="hero"><h1>🚕 Uber Fare Intelligence</h1><p>Fare prediction + interactive business dashboard + model performance</p></div>', unsafe_allow_html=True)

pred_tab, dash_tab, model_tab, about_tab = st.tabs(["🚕 Predict Fare","📊 Business Dashboard","🤖 Model Performance","ℹ️ About"])

with pred_tab:
    st.subheader("Estimate a fare before the trip starts")
    c1,c2,c3=st.columns(3)
    cities=sorted(df['city'].dropna().unique().tolist()); payments=sorted(df['payment_method'].dropna().unique().tolist())
    with c1:
        city=st.selectbox("City",cities,index=0)
        payment=st.selectbox("Payment method",payments,index=0)
        pickup_date=st.date_input("Pickup date", value=df['pickup_time'].max().date())
        pickup_clock=st.time_input("Pickup time")
    city_df=df[df['city']==city]
    with c2:
        st.markdown("**Pickup location**")
        pickup_lat=st.number_input("Pickup latitude",value=float(city_df['pickup_lat'].median()),format="%.6f")
        pickup_lng=st.number_input("Pickup longitude",value=float(city_df['pickup_lng'].median()),format="%.6f")
    with c3:
        st.markdown("**Drop-off location**")
        drop_lat=st.number_input("Drop-off latitude",value=float(city_df['drop_lat'].median()),format="%.6f")
        drop_lng=st.number_input("Drop-off longitude",value=float(city_df['drop_lng'].median()),format="%.6f")

    def haversine(lat1,lon1,lat2,lon2):
        p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
        a=np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return 6371*(2*np.arcsin(np.sqrt(a)))
    calculated=float(haversine(pickup_lat,pickup_lng,drop_lat,drop_lng))
    default_distance=max(0.1, calculated)
    use_manual=st.toggle("Adjust distance manually", value=True, help="The source dataset contains a distance_km field. Adjusting lets you use the same distance definition as training.")
    distance=st.slider("Trip distance (km)",0.1,float(max(30,df['distance_km'].max())),float(min(max(default_distance,0.1),max(30,df['distance_km'].max()))),0.1) if use_manual else default_distance
    dt=pd.Timestamp.combine(pickup_date,pickup_clock)
    row=pd.DataFrame([{
        'city':city,'payment_method':payment,'pickup_lat':pickup_lat,'pickup_lng':pickup_lng,'drop_lat':drop_lat,'drop_lng':drop_lng,
        'distance_km':distance,'pickup_year':dt.year,'pickup_month':dt.month,'pickup_day':dt.day,'pickup_hour':dt.hour,
        'day_of_week':dt.dayofweek,'is_weekend':int(dt.dayofweek>=5),'is_rush_hour':int(dt.hour in [7,8,9,16,17,18,19])
    }])
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Distance",f"{distance:.2f} km"); m2.metric("Pickup hour",f"{dt.hour:02d}:00"); m3.metric("Weekend","Yes" if dt.dayofweek>=5 else "No"); m4.metric("Rush hour","Yes" if dt.hour in [7,8,9,16,17,18,19] else "No")
    if st.button("Predict fare",type="primary",use_container_width=True):
        pred=float(model.predict(row)[0])
        st.success(f"Estimated fare: ${pred:,.2f}")
        st.caption("Prediction is based on the supplied synthetic 50K dataset and should be treated as a portfolio/demo estimate, not a live Uber quote.")

with dash_tab:
    st.subheader("Explore the ride dataset")
    with st.expander("Filters", expanded=True):
        f1,f2,f3=st.columns(3)
        sel_city=f1.multiselect("City",sorted(df.city.unique()),default=sorted(df.city.unique()))
        sel_status=f2.multiselect("Trip status",sorted(df.status.unique()),default=sorted(df.status.unique()))
        sel_pay=f3.multiselect("Payment method",sorted(df.payment_method.unique()),default=sorted(df.payment_method.unique()))
        r1,r2,r3=st.columns(3)
        dist_rng=r1.slider("Distance range (km)",float(df.distance_km.min()),float(df.distance_km.max()),(float(df.distance_km.min()),float(df.distance_km.max())))
        fare_rng=r2.slider("Fare range",float(df.fare_amount.min()),float(df.fare_amount.max()),(float(df.fare_amount.min()),float(df.fare_amount.max())))
        hour_rng=r3.slider("Pickup hour",0,23,(0,23))
        dmin,dmax=df.pickup_time.min().date(),df.pickup_time.max().date()
        date_rng=st.date_input("Pickup date range",value=(dmin,dmax),min_value=dmin,max_value=dmax)
    f=df[df.city.isin(sel_city)&df.status.isin(sel_status)&df.payment_method.isin(sel_pay)&df.distance_km.between(*dist_rng)&df.fare_amount.between(*fare_rng)&df.pickup_hour.between(*hour_rng)].copy()
    if isinstance(date_rng,tuple) and len(date_rng)==2:
        f=f[f.pickup_time.dt.date.between(date_rng[0],date_rng[1])]
    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric("Trips",f"{len(f):,}")
    k2.metric("Avg fare",f"${f.fare_amount.mean():.2f}" if len(f) else "—")
    k3.metric("Avg distance",f"{f.distance_km.mean():.2f} km" if len(f) else "—")
    k4.metric("Completion rate",f"{(f.status.eq('Completed').mean()*100):.1f}%" if len(f) else "—")
    k5.metric("Avg duration",f"{f.trip_duration_min.mean():.1f} min" if len(f) else "—")
    if len(f)==0: st.warning("No trips match the selected filters.")
    else:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(7,4)); ax.hist(f.fare_amount,bins=30); ax.set_title("Fare distribution"); ax.set_xlabel("Fare"); ax.set_ylabel("Trips"); st.pyplot(fig); plt.close(fig)
        with c2:
            fig,ax=plt.subplots(figsize=(7,4)); samp=f.sample(min(5000,len(f)),random_state=42); ax.scatter(samp['distance_km'],samp['fare_amount'],alpha=.4,s=10); ax.set_title("Fare vs distance"); ax.set_xlabel("Distance (km)"); ax.set_ylabel("Fare"); st.pyplot(fig); plt.close(fig)
        c3,c4=st.columns(2)
        with c3:
            g=f.groupby('city',as_index=False).fare_amount.mean().sort_values('fare_amount',ascending=False); fig,ax=plt.subplots(figsize=(7,4)); ax.barh(g['city'],g['fare_amount']); ax.set_title("Average fare by city"); ax.set_xlabel("Average fare"); st.pyplot(fig); plt.close(fig)
        with c4:
            g=f.groupby('pickup_hour',as_index=False).fare_amount.mean(); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(g['pickup_hour'],g['fare_amount'],marker='o'); ax.set_title("Average fare by pickup hour"); ax.set_xlabel("Pickup hour"); ax.set_ylabel("Average fare"); st.pyplot(fig); plt.close(fig)
        st.download_button("Download filtered data",data=f.to_csv(index=False).encode('utf-8'),file_name='uber_filtered_data.csv',mime='text/csv')

with model_tab:
    st.subheader("Model comparison")
    metrics=pd.DataFrame(meta['metrics']).sort_values('RMSE')
    st.dataframe(metrics.style.format({'MAE':'{:.3f}','MSE':'{:.3f}','RMSE':'{:.3f}','R2':'{:.4f}'}),use_container_width=True)
    c1,c2,c3,c4=st.columns(4); best=metrics.iloc[0]
    c1.metric("Selected model",meta['best_model']); c2.metric("MAE",f"{best.MAE:.3f}"); c3.metric("RMSE",f"{best.RMSE:.3f}"); c4.metric("R²",f"{best.R2:.4f}")
    st.info("The model is trained only on Completed trips. Trip IDs, driver IDs, rider IDs, trip status, actual drop time and actual trip duration are excluded to reduce leakage and memorization risk.")
    st.write("**Training features:**", ", ".join(meta['features']))

with about_tab:
    st.subheader("Project notes")
    st.markdown("""
- Source: supplied `uber_trips_dataset_50k.csv` (50,000 rows).
- `passenger_count` is **not present** in the source data, so it was not fabricated for model training.
- The app uses completed rides for fare-model training and keeps cancelled/no-show rides available for dashboard analysis.
- The dataset appears synthetic; city labels and coordinates are not always geographically consistent. Treat outputs as educational/portfolio results.
- Post-trip fields such as actual drop time and actual duration are not used to predict fare before the trip starts.
""")
