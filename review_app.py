"""Local Streamlit review console. No video or measurements leave your machine."""
import json,subprocess,sys
from pathlib import Path
import streamlit as st
from core import resolve_review,apply_overrides
st.set_page_config(page_title='Pothole Lab',layout='wide')
st.title('Pothole Lab · Review')
st.caption('Review provisional tracks, merge duplicates and enter measured depths. Scores are not accuracy probabilities.')
run=Path(st.sidebar.text_input('Run directory','outputs/sample'))
if not (run/'analysis.json').exists():
    st.info('Run pipeline.py analyze first, then enter its output directory.');st.stop()
data=json.loads((run/'analysis.json').read_text());review=json.loads((run/'review.json').read_text())
records=json.loads((run/'potholes.json').read_text())
if (run/'annotated.mp4').exists():st.video(str(run/'annotated.mp4'))
st.dataframe(records,use_container_width=True)
if (run/'merge_suggestions.json').exists():
    with st.expander('Reverse-pass merge suggestions'):
        st.dataframe(json.loads((run/'merge_suggestions.json').read_text()))
ids=list(review['tracks'])
if not ids:st.info('No tracks to review.');st.stop()
tid=st.selectbox('Track',ids,format_func=lambda x:f'P{int(x):03}')
e=review['tracks'][tid];record=next((r for r in records if r['track_id']==tid),None)
if record:
    file=run/'evidence'/f'frame_{record["best_frame"]:06}.jpg'
    if file.exists():st.image(str(file))
with st.form('edit'):
    status=st.selectbox('Status',['candidate','confirmed','rejected','uncertain'],index=['candidate','confirmed','rejected','uncertain'].index(e['status']))
    merge_options=['None']+[i for i in ids if i!=tid]
    merge=st.selectbox('Merge into',merge_options,index=merge_options.index(str(e.get('merge_into'))) if e.get('merge_into') else 0)
    has_depth=st.checkbox('I have a manual depth measurement',value=e.get('depth_cm') is not None)
    depth=st.number_input('Measured depth (cm)',min_value=0.,value=float(e.get('depth_cm') or 0),step=.1)
    notes=st.text_area('Evidence / notes',value=e.get('notes',''))
    save=st.form_submit_button('Save review')
    if save:
        review['tracks'][tid]={'status':status,'merge_into':None if merge=='None' else merge,'depth_cm':depth if has_depth else None,'notes':notes}
        try:
            resolve_review(ids,review)
            (run/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False));st.success('Saved. Rebuild exports to apply changes.')
        except ValueError as ex:st.error(str(ex))
if st.button('Rebuild annotated video and CSV'):
    with st.spinner('Rendering reviewed result...'):
        r=subprocess.run([sys.executable,str(Path(__file__).with_name('pipeline.py')),'export',str(run)],capture_output=True,text=True)
    if r.returncode:st.error(r.stderr)
    else:st.success(r.stdout)
st.divider()
st.subheader('Frame corrections')
st.caption('Edit polygon coordinates or add/remove detections on one frame. These corrections do not propagate automatically. Raw inference is preserved.')
frame_id=st.selectbox('Analyzed frame',[r['frame'] for r in data['frames']])
row=next(r for r in data['frames'] if r['frame']==frame_id)
import cv2
cap=cv2.VideoCapture(data['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,img=cap.read();cap.release()
if ok:st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption=f'Original frame {frame_id}; coordinates use original {data["width"]} × {data["height"]} pixels')
override_path=run/'frame_overrides.json'
overrides=json.loads(override_path.read_text()) if override_path.exists() else {}
with st.form('frame_edit'):
    text=st.text_area('Detection JSON: box [x,y,X,Y], polygon [[x,y],...], score, track_id',json.dumps(overrides.get(str(frame_id),row['detections']),indent=2),height=300)
    if st.form_submit_button('Save frame correction'):
        try:
            new=json.loads(text);overrides[str(frame_id)]=new
            apply_overrides(data,overrides)
            for d in new:
                k=str(d['track_id'])
                if k not in review['tracks']:review['tracks'][k]={'status':'confirmed','merge_into':None,'depth_cm':None,'notes':'Manually added instance'}
            override_path.write_text(json.dumps(overrides,indent=2));(run/'review.json').write_text(json.dumps(review,indent=2));st.success('Saved. Rebuild exports to apply.')
        except (ValueError,TypeError,KeyError) as ex:st.error(str(ex))
