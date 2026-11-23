"""Preprocess coordinates exactly as in training, infer 24 permutations, persist events."""
from pathlib import Path
from datetime import datetime
import sqlite3
import uuid
import joblib
import numpy as np
import pandas as pd
from feature_engineering import build, FEATURES

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'aquawatch.sqlite3'
MODEL = ROOT / 'models' / 'fleeing_vs_combined.joblib'
FEATURE_COLS = [f'ID{i}_F{j}_{feat}' for i in range(1,5) for j in range(10) for feat in FEATURES] + ['Video_Avg_Pairwise_Distance','Video_Std_Pairwise_Distance']
assert len(FEATURE_COLS)==202

def connect():
    con=sqlite3.connect(DB,timeout=30)
    con.execute('''CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, source TEXT NOT NULL,
        video_name TEXT NOT NULL, predicted_label INTEGER, fleeing_probability REAL,
        status TEXT NOT NULL, message TEXT, output_path TEXT)''')
    con.commit()
    return con

def save_event(event):
    with connect() as con:
        con.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)',tuple(event[k] for k in ('id','recorded_at','source','video_name','predicted_label','fleeing_probability','status','message','output_path')))

def history():
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM events ORDER BY recorded_at DESC',con)

def predict_coordinates(csv_path, source='manual', display_name=None, threshold=.5):
    csv_path=Path(csv_path)
    jobid=uuid.uuid4().hex[:12]
    out=ROOT/'outputs'/jobid
    out.mkdir(parents=True,exist_ok=True)
    try:
        master=build(csv_path,out,label='Unknown')
        model=joblib.load(MODEL)
        if getattr(model,'n_features_in_',None)!=202: raise ValueError('Model expects different features')
        if hasattr(model,'feature_names_in_') and list(model.feature_names_in_)!=FEATURE_COLS:
            raise ValueError('Feature names/order differ from training')
        result=[]
        for name,group in master.groupby('Source_Video',sort=True):
            X=group[FEATURE_COLS].astype(float)
            if len(X)!=24 or not np.isfinite(X.to_numpy()).all(): raise ValueError(f'Invalid 24-row features: {name}')
            p=float(model.predict_proba(X.to_numpy())[:,list(model.classes_).index(1)].mean())
            predicted=int(p>=threshold)
            result.append({'Video_Name':name,'Predicted_Label':predicted,'Predicted_Behaviour':'Fleeing' if predicted else 'Non-Fleeing','Fleeing_Probability':p,'Permutation_Count':len(X)})
            save_event({'id':uuid.uuid4().hex,'recorded_at':datetime.now().astimezone().isoformat(timespec='seconds'),
                'source':source,'video_name':name,'predicted_label':predicted,'fleeing_probability':p,
                'status':'completed','message':'Potential stress-related fleeing detected' if predicted else 'No fleeing detected in analysed clip',
                'output_path':str(out)})
        df=pd.DataFrame(result)
        df.to_csv(out/'predictions.csv',index=False)
        return df,out
    except Exception as e:
        save_event({'id':uuid.uuid4().hex,'recorded_at':datetime.now().astimezone().isoformat(timespec='seconds'),
            'source':source,'video_name':display_name or csv_path.name,'predicted_label':None,'fleeing_probability':None,
            'status':'failed','message':str(e),'output_path':str(out)})
        raise

def predict_video(mp4_path,source='manual',threshold=.5):
    from extractor import extract_video
    mp4_path=Path(mp4_path)
    jobdir=ROOT/'outputs'/('video_'+uuid.uuid4().hex[:12]); jobdir.mkdir(parents=True,exist_ok=True)
    coord=jobdir/'coordinates.csv'
    extract_video(mp4_path,coord,ROOT/'weights'/'best_combine.pt',ROOT/'weights'/'bestv7.pt',jobdir/'annotated.mp4')
    return predict_coordinates(coord,source=source,display_name=mp4_path.name,threshold=threshold)
