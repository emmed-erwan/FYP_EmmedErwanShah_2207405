"""Recording-level persistence and provisional, behaviour-based monitoring guidance."""
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import uuid
import pandas as pd
from engine import ROOT

DB = ROOT / 'aquawatch.sqlite3'
# Interface-only thresholds; these are NOT validated biological stress cut-offs.
OCCASIONAL_MAX = 0.20


def connection():
    con = sqlite3.connect(DB, timeout=30)
    con.execute('''CREATE TABLE IF NOT EXISTS recordings (
        id TEXT PRIMARY KEY, analysed_at TEXT NOT NULL, recording_name TEXT NOT NULL,
        pond TEXT NOT NULL, recorded_start TEXT, recorded_date TEXT NOT NULL,
        duration_s REAL, total_segments INTEGER NOT NULL, completed_segments INTEGER NOT NULL,
        fleeing_segments INTEGER NOT NULL, failed_segments INTEGER NOT NULL,
        fleeing_rate REAL, status TEXT NOT NULL, output_path TEXT NOT NULL,
        flagged_offsets TEXT NOT NULL DEFAULT '[]')''')
    con.commit()
    return con


def status_for(positive, completed, failed=0):
    """Return a status and next action without asserting fish health or safety."""
    if completed == 0:
        return ('Analysis unavailable', 'No usable segments were analysed. Check the recording and retry.', 'unknown')
    rate = positive / completed
    if positive == 0:
        title = 'No fleeing detected'
        action = 'Continue routine observation. No immediate action is indicated by the analysed footage.'
        level = 'normal'
    elif rate <= OCCASIONAL_MAX:
        title = 'Occasional fleeing detected'
        action = 'Review flagged footage and continue observing the fish. Investigate disturbances if fleeing recurs.'
        level = 'review'
    else:
        title = 'Frequent fleeing detected'
        action = 'Review flagged recordings promptly and check for possible disturbances or environmental problems. Take corrective action if a problem is confirmed.'
        level = 'attention'
    if failed:
        action += f' {failed} segment(s) could not be analysed; coverage is incomplete.'
    return title, action, level


def save_recording(result, output_dir, pond, recorded_start, analysed_at=None):
    completed = result[result.Status.eq('completed')]
    positive = completed[completed.Prediction.eq('Fleeing')]
    failed = len(result) - len(completed)
    status, _, _ = status_for(len(positive), len(completed), failed)
    started = recorded_start.isoformat(timespec='seconds') if recorded_start else None
    # Unknown recording time: group by analysis date, but do not invent clock times.
    analysed_at = analysed_at or datetime.now().astimezone()
    day = recorded_start.date().isoformat() if recorded_start else analysed_at.date().isoformat()
    offsets = [float(x) for x in positive.Start_s]
    item = {
        'id': uuid.uuid4().hex, 'analysed_at': analysed_at.isoformat(timespec='seconds'),
        'recording_name': str(result.Recording.iloc[0]) if len(result) else 'Unknown',
        'pond': pond.strip(), 'recorded_start': started, 'recorded_date': day,
        'duration_s': float(result.End_s.max()) if len(result) else 0,
        'total_segments': len(result), 'completed_segments': len(completed),
        'fleeing_segments': len(positive), 'failed_segments': failed,
        'fleeing_rate': len(positive)/len(completed) if len(completed) else None,
        'status': status, 'output_path': str(output_dir), 'flagged_offsets': json.dumps(offsets)
    }
    with connection() as con:
        con.execute('''INSERT INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', tuple(item.values()))
    return item


def recordings():
    with connection() as con:
        return pd.read_sql_query('SELECT * FROM recordings ORDER BY analysed_at DESC', con)


def daily_summary(df):
    complete = int(df.completed_segments.sum()) if not df.empty else 0
    positive = int(df.fleeing_segments.sum()) if not df.empty else 0
    failed = int(df.failed_segments.sum()) if not df.empty else 0
    status, action, level = status_for(positive, complete, failed)
    return dict(recordings=len(df), completed=complete, fleeing=positive, failed=failed,
                rate=positive/complete if complete else None, status=status, action=action, level=level)


def flagged_times(record, offsets):
    """Return clock timestamps only when recording start is known; otherwise offsets."""
    start = record.get('recorded_start')
    dt = datetime.fromisoformat(start) if isinstance(start, str) and start else None
    result = []
    for offset in offsets:
        seconds = float(offset)
        if dt:
            result.append((dt + timedelta(seconds=seconds)).strftime('%d %b %Y, %I:%M:%S %p'))
        else:
            m, s = divmod(int(seconds), 60)
            result.append(f'+{m:02d}:{s:02d} into recording')
    return result


def timeline_for(record):
    path = Path(record['output_path'])/'timeline.csv'
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()
