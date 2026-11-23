"""AquaWatch V3.2 — recording-first, actionable Streamlit monitoring UI."""
from datetime import datetime, date, time
from pathlib import Path
import html
import json
import tempfile
import pandas as pd
import streamlit as st
from engine import MODEL, ROOT, predict_coordinates
from recording_pipeline import analyse_recording
from monitoring import recordings, daily_summary, save_recording, status_for, flagged_times, timeline_for

st.set_page_config(page_title='AquaWatch | Monitoring', page_icon='🐟', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp,[data-testid="stAppViewContainer"] {background:#F5F8FA;color:#153448}
[data-testid="stSidebar"] {background:#153448}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] h2 {color:#F4FAFC!important}
.block-container {max-width:1250px;padding-top:1.5rem;padding-bottom:3rem}
h1,h2,h3 {color:#153448}
[data-testid="stMetric"] {background:white;border:1px solid #DCE5EA;border-radius:13px;padding:16px}
[data-testid="stMetricValue"] {color:#153448!important}
[data-testid="stMetricLabel"] p {color:#526575!important}
[data-testid="stFileUploader"] {background:white;border:1px solid #DCE5EA;border-radius:12px;padding:8px}
[data-testid="stFileUploaderDropzone"] {background:#F8FBFC!important;border:1px dashed #7897A4!important}
[data-testid="stFileUploaderDropzone"] * {color:#153448!important}
[data-testid="stFileUploaderDropzone"] button {background:white!important;color:#153448!important}
.stButton>button[kind="primary"] {background:#087F8C!important;color:white!important;border-color:#087F8C!important}
.stButton>button[kind="primary"]:disabled {background:#DCE5EA!important;color:#526575!important}
.aw-hero {padding:25px 28px;background:linear-gradient(110deg,#153448,#087F8C);border-radius:16px;margin-bottom:20px}
.aw-hero h1 {color:white!important;font-size:2rem;margin:0 0 6px}
.aw-hero p {color:#E2F4F5;margin:0}
.aw-status {background:white;border:1px solid #DCE5EA;border-left:6px solid #087F8C;border-radius:12px;padding:18px 20px;margin:12px 0 20px}
.aw-status.review {border-left-color:#B45309}
.aw-status.attention {border-left-color:#B42318}
.aw-status.unknown {border-left-color:#64748B}
.aw-status h3 {margin:0 0 8px;font-size:1.2rem}
.aw-status p {color:#334E60;margin:0}
.aw-muted {color:#526575;font-size:.88rem}
</style>''', unsafe_allow_html=True)

PAGES = ['Daily Dashboard','Analyse Recording','History & Reports','About AquaWatch']
if 'navigation' not in st.session_state:
    st.session_state.navigation = PAGES[0]

def navigate_to_analysis():
    st.session_state.navigation = 'Analyse Recording'

with st.sidebar:
    st.markdown('## 🐟 AquaWatch')
    st.caption('Fish behaviour monitoring · V3.2')
    page = st.radio('Workspace', PAGES, key='navigation', label_visibility='collapsed')
    st.divider()
    st.caption('MODEL STATUS')
    st.write('● Classifier available' if MODEL.is_file() else '⚠ Classifier missing')
    st.caption('Fleeing is a potential stress-related behaviour, not a diagnosis.')


def hero(title, description):
    st.markdown(f'<div class="aw-hero"><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def status_card(positive, completed, failed=0, heading='Monitoring interpretation'):
    title, action, level = status_for(positive, completed, failed)
    rate = positive/completed if completed else None
    detail = f'{positive} of {completed} analysed segments flagged ({rate:.1%}).' if rate is not None else 'No successfully analysed segments.'
    st.markdown(f'<div class="aw-status {level}"><div class="aw-muted">{html.escape(heading)}</div><h3>{html.escape(title)}</h3><p>{html.escape(detail)}<br><strong>Recommended action:</strong> {html.escape(action)}</p></div>', unsafe_allow_html=True)
    st.caption('Status thresholds are provisional interface rules (0%, >0–20%, >20%), not validated biological stress thresholds. Results only represent analysed footage.')


def metrics(summary, daily=False):
    a,b,c,d = st.columns(4)
    a.metric('Recordings' if daily else 'Segments in recording', summary['recordings'] if daily else summary['total_segments'])
    b.metric('Segments analysed', summary['completed_segments'] if not daily else summary['completed'])
    c.metric('Fleeing segments', summary['fleeing_segments'] if not daily else summary['fleeing'])
    d.metric('Fleeing rate', f"{summary['rate']:.1%}" if summary['rate'] is not None else '—')


def recording_row(record, key):
    name = str(record['recording_name'])
    completed = int(record['completed_segments']); positive = int(record['fleeing_segments'])
    failed = int(record['failed_segments'])
    rate = positive/completed if completed else None
    st.markdown(f'**{html.escape(name)}**  ·  {html.escape(str(record["pond"]))}  ·  {html.escape(str(record["recorded_date"]))}')
    a,b,c = st.columns(3)
    a.metric('Analysed segments',completed)
    b.metric('Fleeing segments',positive)
    c.metric('Fleeing rate',f'{rate:.1%}' if rate is not None else '—')
    title, action, level = status_for(positive,completed,failed)
    st.write(f'**Result:** {title}')
    st.write(f'**Next step:** {action}')
    if positive:
        offsets = json.loads(record.get('flagged_offsets') or '[]')
        times = flagged_times(record,offsets)
        st.write('**Fleeing detected at:** ' + ', '.join(times[:8]) + (f' and {len(times)-8} more' if len(times)>8 else ''))
    with st.expander('View segment details and footage', expanded=False):
        detail = timeline_for(record)
        if detail.empty:
            st.info('Segment details are unavailable at the saved location. Recording summary remains in the database.')
        else:
            shown = detail[['Segment','Start_s','End_s','Prediction','Fleeing_Score','Status']].copy()
            shown.columns = ['Segment','Start (s)','End (s)','Behaviour','Fleeing probability','Status']
            st.dataframe(shown, hide_index=True, use_container_width=True)
            st.download_button('Export segment CSV',detail.to_csv(index=False).encode(),file_name=f'{Path(name).stem}_segments.csv',mime='text/csv',key=f'download_{key}')
            flagged = detail[detail.Prediction.eq('Fleeing')]
            choices = flagged if not flagged.empty else detail[detail.Status.eq('completed')]
            if not choices.empty:
                ids = choices.Segment.astype(int).tolist()
                selected = st.selectbox('Review a segment',ids,format_func=lambda x:f'Segment {x}',key=f'segment_{key}')
                chosen = choices[choices.Segment.eq(selected)].iloc[0]
                video = Path(record['output_path'])/'segments'/str(chosen.Clip)
                if video.is_file(): st.video(str(video))
                else: st.info('Segment video file is not available.')
                st.caption(f"Segment offset: {float(chosen.Start_s):.1f}–{float(chosen.End_s):.1f} seconds. Probability: {float(chosen.Fleeing_Score):.1%}.")
    st.divider()


if page == 'Daily Dashboard':
    hero('Daily monitoring', 'Understand today’s analysed recordings and identify what needs attention.')
    all_rec = recordings()
    if all_rec.empty:
        st.info('No recording-level monitoring data yet. Analyse a recording to start your daily dashboard.')
    else:
        ponds = sorted(all_rec.pond.dropna().unique().tolist())
        c1,c2 = st.columns([1,1])
        day = c1.date_input('Monitoring date',value=date.today(),key='daily_date')
        pond = c2.selectbox('Pond / tank',['All ponds']+ponds,key='daily_pond')
        selected = all_rec[all_rec.recorded_date.eq(day.isoformat())].copy()
        if pond != 'All ponds': selected = selected[selected.pond.eq(pond)]
        if pond == 'All ponds' and len(ponds)>1:
            st.info('Results below combine different ponds. Select a pond for a pond-specific status.')
        summary = daily_summary(selected)
        st.subheader('Daily status')
        status_card(summary['fleeing'],summary['completed'],summary['failed'],heading=f'{day:%d %B %Y} · {pond}')
        metrics(summary,daily=True)
        if summary['failed']: st.warning(f"{summary['failed']} segment(s) could not be analysed today.")
        st.subheader('Recordings analysed')
        if selected.empty: st.info('No recordings for the selected date and pond.')
        else:
            for _, r in selected.iterrows(): recording_row(r.to_dict(),f'daily_{r.id}')
    st.button('Analyse a new recording',type='primary',on_click=navigate_to_analysis)
    st.caption('A daily status is calculated from successfully analysed segments, not from continuous surveillance. No footage means unknown status.')

elif page == 'Analyse Recording':
    hero('Analyse recording','Upload footage, receive a clear monitoring result, and expand details only when needed.')
    st.write('**1. Recording information**')
    c1,c2 = st.columns([1,1])
    pond = c1.text_input('Pond / tank name',value='Pond A',help='Used to group recordings for daily monitoring.')
    date_known = c2.checkbox('I know when this recording started',value=True)
    start_dt = None
    if date_known:
        a,b = st.columns(2)
        record_day = a.date_input('Recording date',value=date.today())
        record_time = b.time_input('Recording start time',value=datetime.now().time().replace(second=0,microsecond=0))
        start_dt = datetime.combine(record_day,record_time)
    else:
        st.caption('Only relative offsets (such as +00:32) will be shown. Unknown recording times are grouped by analysis date.')
    uploaded = st.file_uploader('**2. Upload MP4 recording**',type=['mp4'])
    st.caption('Recordings must be at least 6 seconds long. Longer videos are split into approximately 6–10 second segments.')
    if st.button('Start analysis',type='primary',disabled=uploaded is None or not pond.strip()):
        if not MODEL.is_file(): st.error('The Random Forest model file is missing.')
        else:
            progress = st.progress(0,text='Preparing recording...')
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp)/Path(uploaded.name).name
                src.write_bytes(uploaded.getbuffer())
                def update(done,total,message):
                    progress.progress(min(1.0,done/max(total,1)),text=message)
                try:
                    result,out = analyse_recording(src,on_progress=update)
                    # The pipeline uses a temporary input path; use the original filename for persistence.
                    result['Recording'] = uploaded.name
                    result.to_csv(Path(out)/'timeline.csv',index=False)
                    record = save_recording(result,out,pond,start_dt)
                    st.session_state['latest_record_id'] = record['id']
                    st.success('Recording analysis completed and saved to monitoring history.')
                except Exception as exc:
                    st.error(f'Analysis failed: {exc}')
                    st.info('Check that the video is valid, at least six seconds long, and that the model weights are available.')
    if 'latest_record_id' in st.session_state:
        all_rec = recordings()
        found = all_rec[all_rec.id.eq(st.session_state.latest_record_id)]
        if not found.empty:
            st.subheader('3. Your recording result')
            recording_row(found.iloc[0].to_dict(),f'latest_{st.session_state.latest_record_id}')
    with st.expander('Advanced research tool · Coordinates CSV'):
        st.caption('Optional testing feature; farmers can use MP4 upload above.')
        coords = st.file_uploader('Coordinates CSV',type=['csv'],key='coord_upload')
        if st.button('Classify coordinates',disabled=coords is None):
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp)/'coordinates.csv'
                src.write_bytes(coords.getbuffer())
                try:
                    pred,_ = predict_coordinates(src)
                    st.dataframe(pred,hide_index=True,use_container_width=True)
                except Exception as exc: st.error(str(exc))

elif page == 'History & Reports':
    hero('Recording history','Review one summary per recording. Open individual segments only when needed.')
    all_rec = recordings()
    if all_rec.empty: st.info('No recording summaries have been saved yet.')
    else:
        a,b,c = st.columns([2,1,1])
        search = a.text_input('Search recording name')
        ponds = sorted(all_rec.pond.dropna().unique().tolist())
        pond = b.selectbox('Pond / tank',['All ponds']+ponds,key='history_pond')
        day_filter = c.date_input('Date (optional)',value=None,key='history_date')
        selected = all_rec.copy()
        if search: selected = selected[selected.recording_name.str.contains(search,case=False,regex=False,na=False)]
        if pond!='All ponds': selected=selected[selected.pond.eq(pond)]
        if day_filter: selected=selected[selected.recorded_date.eq(day_filter.isoformat())]
        st.caption(f'{len(selected)} recording(s)')
        export = selected[['recorded_date','pond','recording_name','total_segments','completed_segments','fleeing_segments','failed_segments','fleeing_rate','status']].copy()
        st.download_button('Export recording summaries (CSV)',export.to_csv(index=False).encode(),file_name='aquawatch_recordings.csv',mime='text/csv')
        for _, r in selected.iterrows(): recording_row(r.to_dict(),f'history_{r.id}')

else:
    hero('About AquaWatch','A recorded-video prototype supporting review of tilapia behaviour.')
    st.markdown('''**Workflow:** Upload a recording → segment it → detect and track fish with YOLO and BoT-SORT → extract movement features → classify each segment with Random Forest → aggregate results per recording and day.

**Daily status:** Total fleeing-positive segments divided by successfully analysed segments. 0% means no fleeing detected; above 0% to 20% is labelled occasional; above 20% is labelled frequent. These cut-offs are **provisional interface rules**, not validated welfare thresholds.

**Important limitations:** Fleeing is a potential stress-related behaviour indicator, not a diagnosis. Other stress behaviours are not assessed. Uploaded recordings cover only the recorded periods, and missing or failed footage cannot establish normal conditions. The prototype is not continuous or real-time monitoring; open-pond field validation remains future work.

**Time of events:** Clock times are calculated from the user-entered recording start time plus each segment's start offset. Without a known start time, only relative offsets are shown.''')
