import os, sqlite3, uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model.detector import DeepfakeDetector
BASE=os.path.dirname(os.path.abspath(__file__)); UPLOAD=os.path.join(BASE,'uploads'); DB=os.path.join(BASE,'deepfake_history.db')
os.makedirs(UPLOAD,exist_ok=True)
app=Flask(__name__); app.secret_key=os.getenv('SECRET_KEY','change-me'); app.config['MAX_CONTENT_LENGTH']=100*1024*1024
detector=DeepfakeDetector(); ALLOWED={'jpg','jpeg','png','webp','mp4','mov','avi','mkv'}
def init_db():
 con=sqlite3.connect(DB); con.execute('CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY AUTOINCREMENT,filename TEXT,media_type TEXT,prediction TEXT,confidence REAL,details TEXT,created_at TEXT)'); con.commit(); con.close()
@app.route('/')
def index():
 con=sqlite3.connect(DB); total=con.execute('SELECT COUNT(*) FROM scans').fetchone()[0]; fake=con.execute("SELECT COUNT(*) FROM scans WHERE prediction='Potential Deepfake'").fetchone()[0]; real=con.execute("SELECT COUNT(*) FROM scans WHERE prediction='Likely Real'").fetchone()[0]; con.close(); return render_template('index.html',total=total,fake=fake,real=real,model=detector.model_available)
@app.post('/analyze')
def analyze():
 f=request.files.get('media')
 if not f or not f.filename: flash('Select an image or video.'); return redirect(url_for('index'))
 ext=f.filename.rsplit('.',1)[-1].lower() if '.' in f.filename else ''
 if ext not in ALLOWED: flash('Unsupported file type.'); return redirect(url_for('index'))
 name=f'{uuid.uuid4().hex}_{secure_filename(f.filename)}'; path=os.path.join(UPLOAD,name); f.save(path)
 try: result=detector.analyze_video(path) if ext in {'mp4','mov','avi','mkv'} else detector.analyze_image(path); typ='Video' if ext in {'mp4','mov','avi','mkv'} else 'Image'
 except Exception as e:
  if os.path.exists(path): os.remove(path)
  flash(f'Analysis failed: {e}'); return redirect(url_for('index'))
 con=sqlite3.connect(DB); con.execute('INSERT INTO scans(filename,media_type,prediction,confidence,details,created_at) VALUES(?,?,?,?,?,?)',(f.filename,typ,result['prediction'],result['confidence'],result['details'],datetime.now().strftime('%Y-%m-%d %H:%M:%S'))); con.commit(); con.close(); os.remove(path)
 return render_template('result.html',result=result,filename=f.filename,media_type=typ)
@app.route('/history')
def history():
 con=sqlite3.connect(DB); con.row_factory=sqlite3.Row; rows=con.execute('SELECT * FROM scans ORDER BY id DESC LIMIT 100').fetchall(); con.close(); return render_template('history.html',rows=rows)
if __name__=='__main__': init_db(); app.run(debug=True)
