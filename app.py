import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Global Resource Pressure Intelligence", page_icon="🌍", layout="wide")

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp,[data-testid="stAppViewContainer"] {background:#f4f7f5;color:#182b23}
[data-testid="stHeader"] {background:rgba(244,247,245,.96)}
section[data-testid="stSidebar"] {background:#12372a}
section[data-testid="stSidebar"] * {color:#f7fff9!important}
h1,h2,h3,h4,p,li,label,.stMarkdown {color:#20372c}
.stCaption,[data-testid="stCaptionContainer"] p {color:#4d6257!important}
div[data-testid="stMetric"] {background:#fff!important;padding:14px;border-radius:12px;border:1px solid #d4e1d8}
div[data-testid="stMetric"] label,[data-testid="stMetricLabel"] {color:#42574b!important}
[data-testid="stMetricValue"] {color:#163f31!important}
.hero {padding:24px;border-radius:16px;color:#fff;background:linear-gradient(120deg,#12372a,#2e7d5b);margin-bottom:16px}
.hero h1,.hero p {color:#fff!important}
.stTextInput input,.stNumberInput input,[data-baseweb="select"]>div,[data-baseweb="input"]>div {background:#fff!important;color:#172b21!important;border-color:#9aafa1!important}
[data-baseweb="select"] * {color:#172b21!important}
.stButton button,.stDownloadButton button {background:#1f6b4c;color:white;border:1px solid #1f6b4c}
.stButton button:hover,.stDownloadButton button:hover {background:#174f39;color:white;border-color:#174f39}
[data-testid="stAlert"] p {color:#24372c!important}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>Global Resource Pressure & Population Intelligence</h1><p>Explore population trends, predict population growth, classify resource access, and compare machine-learning models.</p></div>', unsafe_allow_html=True)

# ---------- Data preparation ----------
def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    aliases = {
        "country_name":"Country", "Country_Name":"Country", "country":"Country",
        "year":"Year", "region":"Region", "population_total":"Population",
        "Population_Total":"Population", "Population_Growth_Rate":"Population_Growth",
        "Population_Growth_Percent":"Population_Growth", "Population_Density_per_sq_km":"Population_Density",
        "Urban_Population_Percent":"Urban_Population", "Electricity_Access_Percent":"Electricity_Access",
        "Access_to_Electricity":"Electricity_Access", "People_Using_Basic_Drinking_Water_Services":"Water_Access",
        "Water_Access_Percent":"Water_Access"
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old:new})
    return df

def numeric_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pressure_score(frame):
    """Project-defined relative proxy: density/growth increase pressure; access reduces it."""
    density = pd.to_numeric(frame.get("Population_Density", pd.Series(np.nan,index=frame.index)), errors="coerce")
    growth = pd.to_numeric(frame.get("Population_Growth", pd.Series(np.nan,index=frame.index)), errors="coerce")
    water = pd.to_numeric(frame.get("Water_Access", pd.Series(np.nan,index=frame.index)), errors="coerce")
    elec = pd.to_numeric(frame.get("Electricity_Access", pd.Series(np.nan,index=frame.index)), errors="coerce")
    def scaled(s):
        s=s.replace([np.inf,-np.inf],np.nan)
        if s.notna().sum()==0: return pd.Series(0.5,index=frame.index)
        lo,hi=s.quantile(.05),s.quantile(.95)
        if pd.isna(lo) or pd.isna(hi) or hi<=lo: return pd.Series(.5,index=frame.index)
        return ((s.clip(lo,hi)-lo)/(hi-lo)).fillna(.5)
    # Missing indicators are neutral, not assumed to be either good or bad.
    d=scaled(np.log1p(density.clip(lower=0)))
    g=scaled(growth)
    w=(1-water.clip(0,100)/100).where(water.notna(),.5)
    e=(1-elec.clip(0,100)/100).where(elec.notna(),.5)
    return 100*(.30*d+.20*g+.25*w+.25*e)

def prepare_data(raw):
    df=normalize_columns(raw)
    if "Country" not in df.columns or "Year" not in df.columns:
        raise ValueError("Dataset must contain Country and Year columns.")
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"],errors="coerce")
    df=df.dropna(subset=["Country","Year"])
    df["Year"] = df["Year"].astype(int)
    if "Region" not in df.columns: df["Region"]="Unknown"
    df["Region"] = df["Region"].fillna("Unknown").astype(str)
    for c in df.columns:
        if c not in ("Country","Region"):
            df[c]=pd.to_numeric(df[c],errors="ignore")
    df=df.sort_values(["Country","Year"]).drop_duplicates(["Country","Year"],keep="last").reset_index(drop=True)
    if "Population_Growth" not in df.columns:
        popcol=numeric_col(df,["Population","Population_Total"])
        if popcol:
            df["Population_Growth"] = df.groupby("Country")[popcol].pct_change()*100
    if "Electricity_Access" not in df.columns:
        c=numeric_col(df,["Electricity_Access_Percentage","Electricity_Access_Rate"])
        if c: df["Electricity_Access"]=df[c]
    if "Water_Access" not in df.columns:
        c=numeric_col(df,["Water_Access_Percentage","Water_Access_Rate"])
        if c: df["Water_Access"]=df[c]
    # Labels refer to the following calendar year; gaps are not treated as next-year observations.
    next_year=df.groupby("Country")["Year"].shift(-1)
    if "Population_Growth" in df.columns:
        df["Population_Growth_Next_Year"] = df.groupby("Country")["Population_Growth"].shift(-1)
        df.loc[next_year != df["Year"]+1,"Population_Growth_Next_Year"] = np.nan
    else:
        df["Population_Growth_Next_Year"] = np.nan
    for col, target in [("Electricity_Access","Electricity_Access_Level_Next_Year"),("Water_Access","Water_Access_Level_Next_Year")]:
        if col in df.columns:
            future=df.groupby("Country")[col].shift(-1)
            future.loc[next_year != df["Year"]+1]=np.nan
            df[target]=pd.cut(future,[-np.inf,50,80,np.inf],labels=["Low","Moderate","High"],right=False).astype(object)
        else: df[target]=np.nan
    score=pressure_score(df)
    future_score=score.groupby(df["Country"]).shift(-1)
    future_score.loc[next_year != df["Year"]+1]=np.nan
    # Relative categories based on whole dataset score distribution; explicitly a project proxy.
    df["Resource_Pressure_Level_Next_Year"]=pd.cut(future_score,[-np.inf,33.333,66.667,np.inf],labels=["Low","Moderate","High"]).astype(object)
    return df

# Model predictors intentionally exclude the six indicator columns shown in the
# earlier forms (Population, Urban Population, Population Growth, Electricity
# Access, Population Density, and Water Access). Models use only temporal and
# categorical context so these indicators are not exposed or used as predictors.
def feature_columns(df, geography="Country"):
    """Use Year plus exactly one geographic feature: Country OR Region."""
    geography = geography if geography in ("Country", "Region") else "Country"
    return [c for c in ["Year", geography] if c in df.columns]

def make_preprocessor(X):
    cats=[c for c in ["Country","Region"] if c in X.columns]
    nums=[c for c in X.columns if c not in cats]
    transformers=[]
    if nums:
        transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())]),nums))
    if cats:
        transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats))
    return ColumnTransformer(transformers,remainder="drop")

def temporal_split(data):
    data=data.dropna(subset=["Year"]).copy()
    years=sorted(data["Year"].unique())
    if len(years)<3:
        cut=max(1,int(len(data)*.8))
        return data.iloc[:cut],data.iloc[cut:]
    split_year=years[max(0,int(len(years)*.8)-1)]
    train=data[data["Year"]<split_year]
    test=data[data["Year"]>=split_year]
    if train.empty or test.empty:
        split=max(1,int(len(data)*.8)); train=data.iloc[:split]; test=data.iloc[split:]
    return train,test

def regression_models():
    return {"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=180,random_state=42,min_samples_leaf=2),"Gradient Boosting":GradientBoostingRegressor(random_state=42)}
def classification_models():
    return {"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced"),"Decision Tree":DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=42),"Random Forest":RandomForestClassifier(n_estimators=180,class_weight="balanced",random_state=42,min_samples_leaf=2)}

def train_regression(df, model_name, geography="Country"):
    data=df.dropna(subset=["Population_Growth_Next_Year"]).copy()
    if len(data)<10: raise ValueError("Not enough consecutive-year population-growth records to train a model.")
    cols=feature_columns(df, geography); tr,te=temporal_split(data)
    if tr.empty or te.empty: raise ValueError("Could not create train/test split for regression.")
    model=Pipeline([("prep",make_preprocessor(tr[cols])),("model",regression_models()[model_name])])
    model.fit(tr[cols],tr["Population_Growth_Next_Year"])
    pred=model.predict(te[cols]); y=te["Population_Growth_Next_Year"]
    metrics={"MAE":mean_absolute_error(y,pred),"MSE":mean_squared_error(y,pred),"RMSE":mean_squared_error(y,pred)**.5,"R²":r2_score(y,pred) if len(y)>1 else np.nan}
    return model,metrics,te,pred,cols

def train_classifier(df, target_col, model_name, geography="Country"):
    data=df.dropna(subset=[target_col]).copy()
    data=data[data[target_col].astype(str).isin(["Low","Moderate","High"])]
    if len(data)<12 or data[target_col].nunique()<2: raise ValueError("Not enough labelled examples or classes for this target.")
    cols=feature_columns(df, geography); tr,te=temporal_split(data)
    if tr.empty or te.empty: raise ValueError("Could not create train/test split for classification.")
    # Remove classes not represented in training so model training remains valid.
    seen=set(tr[target_col].astype(str)); te=te[te[target_col].astype(str).isin(seen)]
    if te.empty: raise ValueError("Test split contains no classes seen during training; use more years of data.")
    model=Pipeline([("prep",make_preprocessor(tr[cols])),("model",classification_models()[model_name])])
    model.fit(tr[cols],tr[target_col].astype(str))
    pred=model.predict(te[cols]); y=te[target_col].astype(str)
    metrics={"Accuracy":accuracy_score(y,pred),"Precision weighted":precision_score(y,pred,average="weighted",zero_division=0),"Recall weighted":recall_score(y,pred,average="weighted",zero_division=0),"F1 weighted":f1_score(y,pred,average="weighted",zero_division=0)}
    return model,metrics,te,pred,cols

def fmt_pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "N/A"

# ---------- Load data ----------
with st.sidebar:
    st.title("Navigation")
    page=st.radio("Go to",["Home","Regression Model","Classification Models","Data & Insights"])
    st.divider()
    st.subheader("Dataset")
    uploaded=st.file_uploader("Upload cleaned CSV",type=["csv"])
    st.caption("Required columns: Country and Year. Indicators improve model usefulness.")

if uploaded is not None:
    try: raw_df=pd.read_csv(uploaded)
    except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
else:
    local_path="/content/population_resources_cleaned.csv"
    try: raw_df=pd.read_csv(local_path)
    except Exception:
        st.info("Upload your cleaned dataset CSV using the sidebar to start.")
        st.stop()
try: df=prepare_data(raw_df)
except Exception as e: st.error(f"Dataset problem: {e}"); st.stop()

country_options=sorted(df["Country"].dropna().astype(str).unique())
region_options=sorted(df["Region"].dropna().astype(str).unique())
latest_year=int(df["Year"].max())

# ---------- HOME ----------
if page=="Home":
    st.subheader("Project overview")
    st.write("This application explores how population growth relates to pressure on water, electricity and other resources worldwide. It combines historical data analysis with machine-learning predictions.")
    a,b,c,d=st.columns(4)
    a.metric("Countries",f"{df['Country'].nunique():,}")
    b.metric("Regions",f"{df['Region'].nunique():,}")
    c.metric("Year coverage",f"{int(df.Year.min())}–{latest_year}")
    popcol=numeric_col(df,["Population"])
    d.metric("Population records",f"{df[popcol].notna().sum():,}" if popcol else "N/A")
    st.markdown("### Global trends")
    cols=[c for c in ["Population_Growth","Electricity_Access","Water_Access","Population_Density"] if c in df.columns]
    if cols:
        selected=st.selectbox("Indicator",cols)
        yearly=df.groupby("Year",as_index=False)[selected].mean(numeric_only=True)
        st.plotly_chart(px.line(yearly,x="Year",y=selected,markers=True,title=f"Average {selected.replace('_',' ')} over time"),use_container_width=True)
    st.markdown("### How to use the app")
    st.write("1. Explore a population-growth prediction in Regression Model.  2. Choose a country and scenario year in Classification Models.  3. Review data quality and model evaluation in Data & Insights.")

# ---------- REGRESSION ----------
elif page=="Regression Model":
    st.subheader("Population growth prediction")
    st.write("Train a regression model on historical next-year population growth, then explore a country or region scenario.")
    st.caption("The six indicator values are excluded from predictors. Choose either Country or Region as the single geographic input; both are never used together.")
    geography=st.radio("Prediction level",["Country","Region"],horizontal=True,key="reg_geography")
    model_name=st.selectbox("Regression algorithm",list(regression_models().keys()))
    if st.button("Train regression model",type="primary"):
        try:
            model,metrics,test,pred,cols=train_regression(df,model_name,geography)
            st.session_state["reg_model"]=model; st.session_state["reg_metrics"]=metrics; st.session_state["reg_name"]=model_name; st.session_state["reg_features"]=cols; st.session_state["reg_model_geography"]=geography
            st.session_state["reg_test_plot"]=pd.DataFrame({"Actual":test["Population_Growth_Next_Year"].values,"Predicted":pred})
            st.success(f"{model_name} trained successfully.")
        except Exception as e: st.error(str(e))
    if "reg_model" in st.session_state:
        st.markdown(f"### {st.session_state['reg_name']} test performance")
        m=st.session_state["reg_metrics"]; a,b,c,d=st.columns(4)
        a.metric("MAE",f"{m['MAE']:.3f}"); b.metric("MSE",f"{m['MSE']:.3f}"); c.metric("RMSE",f"{m['RMSE']:.3f}"); d.metric("R²",f"{m['R²']:.3f}" if pd.notna(m['R²']) else "N/A")
        if "reg_test_plot" in st.session_state and not st.session_state["reg_test_plot"].empty:
            st.plotly_chart(px.scatter(st.session_state["reg_test_plot"],x="Actual",y="Predicted",title="Actual vs predicted population growth"),use_container_width=True)
        else:
            st.info("Train the regression model again to generate the actual-versus-predicted chart.")
        st.divider(); st.markdown("### Make a prediction")
        prediction_level=st.session_state.get("reg_model_geography", geography)
        if geography != prediction_level:
            st.warning(f"The trained model uses {prediction_level}. Click **Train regression model** again to switch to {geography}.")
        with st.form("reg_prediction_form"):
            c1,c2=st.columns(2)
            with c1:
                if prediction_level=="Country":
                    selected_geo=st.selectbox("Country",country_options,key="reg_prediction_country")
                else:
                    selected_geo=st.selectbox("Region",region_options,key="reg_prediction_region")
            with c2:
                year=st.number_input("Scenario input year",min_value=1900,max_value=2100,value=latest_year,step=1,key="reg_year")
            run=st.form_submit_button("Predict next-year population growth",type="primary",disabled=(geography != prediction_level))
        if run:
            row={"Year":int(year),prediction_level:selected_geo}
            model_input=pd.DataFrame([row])[st.session_state["reg_features"]]
            result=float(st.session_state["reg_model"].predict(model_input)[0])
            st.metric(f"Estimated population growth in {int(year)+1}",f"{result:.3f}%")
            st.write(f"**Prediction level:** {prediction_level}\n\n**Selected {prediction_level.lower()}:** {selected_geo}")
            st.info("This is a model-based estimate using only the selected geographic category and year. It is not an official population forecast.")
    else: st.info("Choose a regression algorithm and click Train regression model to enable prediction.")

# ---------- CLASSIFICATION ----------
elif page=="Classification Models":
    st.subheader("Resource access and pressure classification")
    st.caption("The six indicator values are excluded from predictors. Choose either Country or Region as the single geographic input; both are never used together.")
    geography=st.radio("Prediction level",["Country","Region"],horizontal=True,key="class_geography")
    targets={"Electricity access level":"Electricity_Access_Level_Next_Year","Water access level":"Water_Access_Level_Next_Year","Resource pressure level":"Resource_Pressure_Level_Next_Year"}
    target_label=st.selectbox("What would you like to classify?",list(targets.keys()))
    target_col=targets[target_label]
    model_name=st.selectbox("Classification algorithm",list(classification_models().keys()))
    if st.button("Train classification model",type="primary"):
        try:
            model,metrics,test,pred,cols=train_classifier(df,target_col,model_name,geography)
            st.session_state["class_model"]=model; st.session_state["class_metrics"]=metrics; st.session_state["class_name"]=model_name; st.session_state["class_target_label"]=target_label; st.session_state["class_target_col"]=target_col; st.session_state["class_features"]=cols; st.session_state["class_model_geography"]=geography
            st.success(f"{model_name} trained successfully for {target_label}.")
        except Exception as e: st.error(str(e))
    if "class_model" in st.session_state:
        st.markdown(f"### {st.session_state['class_name']} test performance — {st.session_state['class_target_label']}")
        m=st.session_state["class_metrics"]; a,b,c,d=st.columns(4)
        a.metric("Accuracy",fmt_pct(m["Accuracy"]*100)); b.metric("Precision",fmt_pct(m["Precision weighted"]*100)); c.metric("Recall",fmt_pct(m["Recall weighted"]*100)); d.metric("F1-score",fmt_pct(m["F1 weighted"]*100))
        prediction_level=st.session_state.get("class_model_geography", geography)
        if geography != prediction_level:
            st.warning(f"The trained model uses {prediction_level}. Click **Train classification model** again to switch to {geography}.")
        st.divider(); st.markdown("### Make a classification prediction")
        st.caption("For years beyond the dataset, this is a scenario based on the selected country or region and year. The six indicator values are not model inputs.")
        with st.form("class_prediction_form"):
            c1,c2=st.columns(2)
            with c1:
                if prediction_level=="Country":
                    selected_geo=st.selectbox("Country",country_options,key="class_prediction_country")
                else:
                    selected_geo=st.selectbox("Region",region_options,key="class_prediction_region")
            with c2:
                year=st.number_input("Scenario input year",min_value=1900,max_value=2050,value=latest_year,step=1,key="class_year")
            run=st.form_submit_button("Predict selected category",type="primary",disabled=(geography != prediction_level))
        if run:
            row={"Year":int(year),prediction_level:selected_geo}
            model_input=pd.DataFrame([row])[st.session_state["class_features"]]
            predicted=str(st.session_state["class_model"].predict(model_input)[0])
            target_year=int(year)+1
            st.markdown("### Prediction result")
            st.metric("Predicted category",predicted)
            st.write(f"**Prediction level:** {prediction_level}\n\n**Selected {prediction_level.lower()}:** {selected_geo}\n\n**Predicted target year:** {target_year}")
            if target_label in ["Electricity access level","Water access level"]:
                resource="electricity access" if target_label=="Electricity access level" else "water access"
                interpretation={"Low":f"The selected {prediction_level.lower()} ({selected_geo}) is predicted to be in the **Low** {resource} category for {target_year}, corresponding to below 50% access in the historical labels.","Moderate":f"The selected {prediction_level.lower()} ({selected_geo}) is predicted to be in the **Moderate** {resource} category for {target_year}, corresponding to 50% to below 80% access in the historical labels.","High":f"The selected {prediction_level.lower()} ({selected_geo}) is predicted to be in the **High** {resource} category for {target_year}, corresponding to 80% access or above in the historical labels."}
                st.info(interpretation.get(predicted,"Interpretation is unavailable for this label."))
                st.caption("This is a predicted category, not a precise percentage estimate or a guarantee of future access.")
            else:
                interpretation={"Low":"Lower relative resource-pressure category.","Moderate":"Middle relative resource-pressure category.","High":"Higher relative resource-pressure category."}
                st.info(interpretation.get(predicted,"This is a project-defined resource-pressure category."))
                st.caption("Resource pressure is a project-defined proxy combining population density, growth, and limited access to water/electricity. It is not an official World Bank rating.")
    else: st.info("Choose the target and algorithm, then train the model to enable single-country predictions.")

# ---------- DATA & INSIGHTS ----------
else:
    st.subheader("Data & Insights")
    st.write("Explore the dataset's coverage, population and resource-access trends, country comparisons, data quality, and model performance.")

    # Dataset overview
    a,b,c,d=st.columns(4)
    a.metric("Records",f"{len(df):,}")
    b.metric("Countries",f"{df.Country.nunique():,}")
    c.metric("Regions",f"{df.Region.nunique():,}")
    d.metric("Year coverage",f"{int(df.Year.min())}–{latest_year}")

    indicator_options=[c for c in ["Population","Population_Growth","Population_Density","Urban_Population","Electricity_Access","Water_Access"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    st.markdown("### Global indicator trends")
    if indicator_options:
        trend_indicator=st.selectbox("Choose an indicator",indicator_options,key="insight_trend_indicator")
        trend_method=st.radio("Aggregation",["Global total (sum)","Country average (mean)","Median country value"],horizontal=True,key="insight_trend_aggregation")
        if trend_method=="Global total (sum)":
            trend=df.groupby("Year",as_index=False)[trend_indicator].sum(min_count=1)
            y_label=f"Total {trend_indicator.replace('_',' ')}"
        elif trend_method=="Median country value":
            trend=df.groupby("Year",as_index=False)[trend_indicator].median()
            y_label=f"Median {trend_indicator.replace('_',' ')}"
        else:
            trend=df.groupby("Year",as_index=False)[trend_indicator].mean()
            y_label=f"Average {trend_indicator.replace('_',' ')}"
        st.plotly_chart(px.line(trend,x="Year",y=trend_indicator,markers=True,title=y_label),use_container_width=True)
        st.caption("Global totals are suitable for population counts. Percentages and rates are usually easier to interpret using country averages or medians.")
    else:
        st.info("No numeric indicator columns were found beyond Country and Year.")

    st.markdown("### Compare countries in a selected year")
    if indicator_options:
        c1,c2=st.columns([1,2])
        with c1:
            compare_year=st.selectbox("Year",sorted(df.Year.unique(),reverse=True),key="insight_compare_year")
            compare_indicator=st.selectbox("Indicator to compare",indicator_options,key="insight_compare_indicator")
            top_n=st.slider("Number of countries",5,30,10,key="insight_top_n")
        year_data=df[df.Year==compare_year][["Country","Region",compare_indicator]].dropna()
        if not year_data.empty:
            with c2:
                ranked=year_data.sort_values(compare_indicator,ascending=False).head(top_n)
                st.plotly_chart(px.bar(ranked.sort_values(compare_indicator),x=compare_indicator,y="Country",color="Region",orientation="h",title=f"Highest {compare_indicator.replace('_',' ')} values in {compare_year}"),use_container_width=True)
            st.dataframe(year_data.sort_values(compare_indicator,ascending=False),use_container_width=True,hide_index=True)
        else:
            st.info("No data is available for this indicator and year.")

    st.markdown("### Regional trends")
    regional_options=[c for c in ["Population_Growth","Urban_Population","Electricity_Access","Water_Access","Population_Density"] if c in indicator_options]
    if regional_options:
        regional_indicator=st.selectbox("Regional indicator",regional_options,key="insight_region_indicator")
        regional=df.groupby(["Year","Region"],as_index=False)[regional_indicator].mean()
        st.plotly_chart(px.line(regional,x="Year",y=regional_indicator,color="Region",title=f"Average {regional_indicator.replace('_',' ')} by region",markers=True),use_container_width=True)
    else:
        st.info("Regional trend chart requires numeric growth, density, urban-population, electricity-access, or water-access columns.")

    st.markdown("### Indicator relationships")
    numeric_df=df[indicator_options].apply(pd.to_numeric,errors="coerce") if indicator_options else pd.DataFrame()
    if len(indicator_options)>=2:
        corr=numeric_df.corr()
        st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Correlation between available indicators"),use_container_width=True)
        st.caption("Correlation measures association, not causation. Relationships can also reflect missing data, country differences, and changes over time.")
    else:
        st.info("At least two numeric indicators are needed to calculate correlations.")

    st.markdown("### Data quality")
    missing=(df.isna().mean()*100).sort_values(ascending=False).rename("Missing (%)").reset_index()
    missing.columns=["Column","Missing (%)"]
    q1,q2=st.columns([2,1])
    with q1:
        st.plotly_chart(px.bar(missing[missing["Missing (%)"]>0],x="Column",y="Missing (%)",title="Missing values by column (%)"),use_container_width=True)
    with q2:
        st.metric("Duplicate country-year rows",int(df.duplicated(["Country","Year"]).sum()))
        st.metric("Cells missing",f"{int(df.isna().sum().sum()):,}")
        st.metric("Total cells",f"{df.shape[0]*df.shape[1]:,}")
    with st.expander("View missing-value summary"):
        st.dataframe(missing,use_container_width=True,hide_index=True)

    st.markdown("### Dataset preview and download")
    preview_rows=st.slider("Rows to preview",min(10,len(df)),min(500,len(df)),min(100,len(df)),key="insight_preview_rows")
    st.dataframe(df.head(preview_rows),use_container_width=True,hide_index=True)
    st.download_button("Download prepared dataset",df.to_csv(index=False),"prepared_resource_population_data.csv","text/csv")

    st.markdown("### Machine-learning model evaluation")
    st.write("Compare algorithms using the app's time-ordered train/test split. Indicator values are not model inputs.")
    benchmark_geography=st.radio("Geographic feature for benchmark",["Country","Region"],horizontal=True,key="benchmark_geography")
    st.caption(f"Benchmark predictors: Year + {benchmark_geography} only.")
    targets={"Electricity access level":"Electricity_Access_Level_Next_Year","Water access level":"Water_Access_Level_Next_Year","Resource pressure level":"Resource_Pressure_Level_Next_Year"}
    if st.button("Run model benchmark",type="primary"):
        reg_results=[]; cls_results=[]
        for name in regression_models():
            try:
                _,metrics,_,_,_=train_regression(df,name,benchmark_geography)
                reg_results.append({"Algorithm":name,**metrics})
            except Exception as e: st.warning(f"Regression — {name}: {e}")
        for target_label,target_col in targets.items():
            for name in classification_models():
                try:
                    _,metrics,_,_,_=train_classifier(df,target_col,name,benchmark_geography)
                    cls_results.append({"Target":target_label,"Algorithm":name,**metrics})
                except Exception as e: st.warning(f"Classification — {target_label} / {name}: {e}")
        st.session_state["bench_reg"]=pd.DataFrame(reg_results)
        st.session_state["bench_cls"]=pd.DataFrame(cls_results)
    if "bench_reg" in st.session_state:
        st.markdown("#### Regression algorithms")
        br=st.session_state["bench_reg"]
        if br.empty: st.warning("No regression algorithm could be evaluated.")
        else:
            st.dataframe(br.sort_values("RMSE"),use_container_width=True,hide_index=True)
            st.caption("Lower MAE, MSE and RMSE are better; R² closer to 1 generally indicates stronger predictive fit.")
            st.plotly_chart(px.bar(br.melt(id_vars="Algorithm",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error"),x="Algorithm",y="Error",color="Metric",barmode="group",title="Regression error comparison (lower is better)"),use_container_width=True)
        st.markdown("#### Classification algorithms")
        bc=st.session_state["bench_cls"]
        if bc.empty: st.warning("No classification algorithm could be evaluated.")
        else:
            st.dataframe(bc.sort_values("F1 weighted",ascending=False),use_container_width=True,hide_index=True)
            st.caption("Higher accuracy, weighted precision, weighted recall and weighted F1-score are better. Compare models within the same target.")
            st.plotly_chart(px.bar(bc,x="Algorithm",y="F1 weighted",color="Target",barmode="group",title="Classification weighted F1-score (higher is better)"),use_container_width=True)
        st.download_button("Download regression evaluation metrics",st.session_state["bench_reg"].to_csv(index=False),"regression_metrics.csv","text/csv")
        st.download_button("Download classification evaluation metrics",st.session_state["bench_cls"].to_csv(index=False),"classification_metrics.csv","text/csv")
    else:
        st.info("Click Run model benchmark to calculate evaluation metrics for each algorithm.")
