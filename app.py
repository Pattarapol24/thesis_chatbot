from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, flash, session  
# render_template = เปิดหน้าเว็บ request = รับข้อมูลจากผู้ใช้ redirect = เปลี่ยนหน้าเว็บ url_for = สร้าง URL jsonify = ส่งJson 
# send_from_directory = ส่งไฟล์จากโฟลเดอร์ flash = เก็บข้อความชั่วคราว session = เก็บข้อมูลผู้ใช้ชั่วคราว
from flask_sqlalchemy import SQLAlchemy  
import os, json, re, uuid  # os = จัดการไฟล์/โฟลเดอร์  # json = แปลงข้อมูลระหว่าง JSON กับ Python  
# re = โมดูลจัดการ/ค้นหาข้อความตามรูปแบบ  # uuid = เข้ารหัส
from config import UPLOAD_FOLDER, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI  
from utils.pdf_utils import extract_text_from_pdf  
from utils.vector_utils import add_doc_to_vectorstore, search_similar, search_Ranking
from utils.intent_classifier import classify_intent 
from api.typhoon_api import ask_typhoon  
from sqlalchemy import or_ 
from utils.vector_utils import VECTOR_DB  
from requests_oauthlib import OAuth2Session  
import pickle  # บันทึก vector map_id เป็นไฟล์
import faiss  

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  
app.secret_key = os.urandom(24)  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # ปิดการติดตามการเปลี่ยนแปลงเพื่อลดการโหลด
db = SQLAlchemy(app)  # สร้างอ็อบเจ็กต์จัดการฐานข้อมูล
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # อนุญาตให้ใช้ HTTP  ในการพัฒนา

# ---------------------------
# OAuth2 Config
# ---------------------------
AUTHORIZATION_BASE_URL = 'https://accounts.google.com/o/oauth2/auth'  
TOKEN_URL = 'https://oauth2.googleapis.com/token'  
SCOPE = ['https://www.googleapis.com/auth/userinfo.email',  
         'https://www.googleapis.com/auth/userinfo.profile']  

# ----------------------------
# DB Models
# ----------------------------
class Thesis(db.Model):  
    id = db.Column(db.Integer, primary_key=True)  
    title = db.Column(db.String)  
    author = db.Column(db.String) 
    advisor = db.Column(db.String)  
    year = db.Column(db.String) 
    filename = db.Column(db.String)  
    text = db.Column(db.Text) 

class Lecturer(db.Model):  
    id = db.Column(db.Integer, primary_key=True) 
    title = db.Column(db.String, nullable=True) 
    name = db.Column(db.String, nullable=False)  
    department = db.Column(db.String, nullable=False) 
    expertise = db.Column(db.Text, nullable=False)  
    link = db.Column(db.String, nullable=True)  
    def __repr__(self):  # ฟังก์ชันแสดงผลอ็อบเจ็กต์เมื่อ print()
        full_name = f"{self.title} {self.name}" if self.title else self.name  
        return f"<Lecturer {full_name}>"  

with app.app_context():  # เปิด context ของแอป Flask เพื่อให้เข้าถึงฐานข้อมูลได้
    db.create_all()  # สร้างตารางทั้งหมดในฐานข้อมูล ถ้ายังไม่มีอยู่แล้ว

# ----------------------------
# สร้าง FAISS index ว่างถ้ายังไม่มี
# ----------------------------

EMBEDDING_SIZE = 384  

os.makedirs(os.path.dirname(VECTOR_DB), exist_ok=True)  # สร้างโฟลเดอร์สำหรับเก็บไฟล์ฐานข้อมูลเวกเตอร์ หากยังไม่มี
if not os.path.exists(VECTOR_DB):  # ตรวจสอบว่าไฟล์ฐานข้อมูลเวกเตอร์มีอยู่แล้วหรือไม่
    faiss_db = faiss.IndexFlatL2(EMBEDDING_SIZE)  # สร้างฐานข้อมูลเวกเตอร์ FAISS แบบ L2 
    id_map = {}  
    with open(VECTOR_DB, 'wb') as f:  
        pickle.dump((faiss_db, id_map), f)  # บันทึกอ็อบเจ็กต์ FAISS และ id_map ลงไฟล์
    print(f"[INFO] สร้างไฟล์ FAISS index ว่างที่ {VECTOR_DB}")  
else:  
    print(f"[INFO] ไฟล์ FAISS index มีอยู่แล้ว") 


# ----------------------------
# Utility Functions
# ----------------------------
def extract_year_from_pdf(text: str) -> str:  
    match = re.search(r"ปีการศึกษา\s*[:\-]?\s*(25\d{2}|20\d{2})", text)  # ใช้ re หา "ปีการศึกษา" ตามด้วยตัวเลข
    if match:  
        year = int(match.group(1))  # แปลงค่าปีจากข้อความเป็นตัวเลข 
        return str(year + 543) if 2000 <= year <= 2099 else str(year)  
    return "ไม่ระบุ"  

def extract_year(text: str) -> str | None:  
    match = re.search(r"(20\d{2}|25\d{2})", text)  # ค้นหาตัวเลขปีที่อยู่ในช่วง 2000–2599
    print("match",match)
    if match:  
        year = int(match.group(1))  
        return str(year + 543) if 2000 <= year <= 2099 else str(year)  # คืนค่าปีในรูปแบบข้อความ
    return None  

def extract_keyword(text: str):  
    keyword_prompt = f"""
จากข้อความนี้เท่านั้น: "{text}"
จงสกัดคำค้นหลักที่เกี่ยวข้องกับหัวข้อปริญญานิพนธ์เท่านั้น
- ห้ามส่งคำทั่วไปหรือคำที่เกี่ยวกับปี, โปรเจค, ปริญญานิพนธ์, เรื่อง, หัวข้อ, การศึกษา, งานวิจัย
- ให้ตอบเพียงคำค้นหลัก 1 คำ หรือวลีสั้น ๆ เท่านั้น
- หากข้อความนี้ไม่มีคำค้นเฉพาะ ให้ตอบเป็น None
"""  
    print("ข้อความที่ส่งให้typhoonสกัด",text)  
    try:
        main_keyword = ask_typhoon(system_prompt=keyword_prompt).strip()  
        if main_keyword.lower() == "none":  
            main_keyword = None  
    except Exception as e:  
        print("⚠️ extract keyword error:", e)  
        main_keyword = None  

    if main_keyword:  
        expansion_prompt = f"""
จากคำค้น "{main_keyword}"
จงให้คำที่มีความหมายใกล้เคียงหรือเกี่ยวข้องกัน (ภาษาไทยหรืออังกฤษ)
เช่น คำพ้องความหมาย หรือคำที่มักใช้แทนกัน
**ห้ามรวมตัวเลขปี หรือคำที่เกี่ยวกับปี เช่น 'ปี', 'พ.ศ.', 'ค.ศ.'**
ให้ตอบเป็นรายการคั่นด้วยเครื่องหมายจุลภาค (,)
และไม่เกิน 5 คำ
ตัวอย่าง:
AI -> ปัญญาประดิษฐ์, Artificial Intelligence, machine learning, deep learning
"""  
        try:
            related_text = ask_typhoon(system_prompt=expansion_prompt).strip()
            print("related_text",related_text)
            related_keywords = [k.strip() for k in re.split(r"[,、;]", related_text) if k.strip()]  # แยกรายการคำออกจากกันด้วยจุลภาค
            print("related_keywords",related_keywords)
        except Exception as e:  
            print("⚠️ expand keyword error:", e)  
            related_keywords = []  
    else:  # ถ้าไม่มีคำหลัก
        related_keywords = [] 

    return main_keyword, related_keywords 


def extract_title(text: str) -> str:  
    prompt = f"สกัดชื่อเรื่องจากประโยคนี้: \"{text}\" ตอบเฉพาะชื่อเรื่อง" 
    try:
        return ask_typhoon(system_prompt=prompt).strip() 
    except:
        return text  

def log_debug(intent: str, data: dict):  # ฟังก์ชันพิมพ์ข้อมูลดีบัก
    print(f"[DEBUG] Intent: {intent}") 
    for k, v in data.items():  # วนลูปแสดง key และ value ใน dict
        print(f"        {k}: {v}")  # แสดงข้อมูลในรูปแบบอ่านง่าย

# ----------------------------
# Admin Routes (CRUD Lecturers)
# ----------------------------
@app.route("/admin/lecturers") 
def admin_lecturers():
    lecturers = Lecturer.query.all()  # ดึงข้อมูลอาจารย์ทั้งหมดจากฐานข้อมูล
    return render_template("lecturers_list.html", lecturers=lecturers)  

@app.route("/admin/lecturers/add", methods=["GET", "POST"])  
def add_lecturer():
    titles = ["", "ผศ.", "ผศ.ดร.", "รศ.ดร.", "อ.ดร.", "อ."]  # รายการคำนำหน้าใน dropdown
    if request.method == "POST":  
        title = request.form.get("title", "")  
        name = request.form["name"]  
        department = request.form["department"]  
        expertise = request.form["expertise"]  
        link = request.form.get("link")  
        db.session.add(Lecturer(  
            title=title,
            name=name,
            department=department,
            expertise=expertise,
            link=link
        ))
        db.session.commit()  
        flash("เพิ่มอาจารย์เรียบร้อยแล้ว", "success")  # แสดงข้อความแจ้งเตือนว่าทำสำเร็จ
        return redirect(url_for("admin_lecturers"))  
    return render_template("lecturers_form.html", action="add", titles=titles)  


@app.route("/admin/lecturers/edit/<int:id>", methods=["GET", "POST"])  
def edit_lecturer(id):
    lecturer = Lecturer.query.get_or_404(id)  
    titles = ["", "ผศ.", "ผศ.ดร.", "รศ.ดร.", "อ.ดร.", "อ."]  
    if request.method == "POST":  
        lecturer.title = request.form.get("title", "")   # อัปเดต
        lecturer.name = request.form["name"]  
        lecturer.department = request.form["department"]  
        lecturer.expertise = request.form["expertise"]  
        lecturer.link = request.form.get("link")  
        db.session.commit()  
        flash("แก้ไขข้อมูลอาจารย์เรียบร้อยแล้ว", "success")  
        return redirect(url_for("admin_lecturers"))  
    return render_template("lecturers_form.html", action="edit", lecturer=lecturer, titles=titles) 


@app.route("/admin/lecturers/delete/<int:id>", methods=["POST"])  
def delete_lecturer(id):
    lecturer = Lecturer.query.get_or_404(id)  
    db.session.delete(lecturer) 
    db.session.commit()  
    flash("ลบอาจารย์เรียบร้อยแล้ว", "success")  
    return redirect(url_for("admin_lecturers"))  

# ----------------------------  
# Upload PDFs
# ----------------------------
@app.route('/admin/upload', methods=['GET','POST']) 
def admin_upload():  
    if request.method == 'POST':  
        files = request.files.getlist('pdf_files')  # ดึงรายการไฟล์ PDF ทั้งหมดจากฟอร์ม
        meta_datas = json.loads(request.form.get('meta_datas'))  # โหลดข้อมูลเมทาดาทาที่ส่งมาจากฟอร์ม (เป็น JSON)

        if len(files) != len(meta_datas): 
            flash("จำนวน meta data ไม่ตรงกับไฟล์ กรุณาลองใหม่", "error")  
            return redirect(url_for('admin_upload'))

        allowed_ext = {'.pdf'}  
        duplicated_files = []  # สร้างลิสต์เก็บชื่อไฟล์ที่ซ้ำ

        for i, file in enumerate(files):  # วนลูปไฟล์ที่อัปโหลดเข้ามาทั้งหมด enumerateวนลูปให้หมายเลข
            if file:  # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
                filename = file.filename  
                ext = os.path.splitext(filename)[1].lower()  # แยกนามสกุลไฟล์และแปลงเป็นตัวพิมพ์เล็ก
                if ext not in allowed_ext:  # ตรวจสอบว่านามสกุลไฟล์อยู่ใน allowed_ext หรือไม่
                    flash(f"ไฟล์ {filename} ไม่ใช่ไฟล์ PDF ที่อนุญาต", "error") 
                    return redirect(url_for('admin_upload'))  

                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # สร้าง path สำหรับบันทึกไฟล์
                if os.path.exists(save_path):  # ตรวจสอบว่ามีไฟล์ชื่อซ้ำในโฟลเดอร์หรือไม่
                    duplicated_files.append(filename)  # ถ้ามีซ้ำให้เพิ่มชื่อไฟล์ในลิสต์ duplicated_files
                    continue  

                file.save(save_path) 
                pdf_text = extract_text_from_pdf(save_path) 
                year_from_pdf = extract_year_from_pdf(pdf_text)  
                meta = meta_datas[i]  # ดึงเมทาดาทาที่ตรงกับไฟล์นี้จากลิสต์
                thesis = Thesis(title=meta['title'], author=meta['author'],  # สร้างอ็อบเจกต์ Thesis เพื่อเก็บข้อมูลลงฐานข้อมูล
                                advisor=meta['advisor'], year=year_from_pdf,
                                filename=filename, text=pdf_text)
                db.session.add(thesis) 
                db.session.commit()  
                add_doc_to_vectorstore(str(thesis.id), pdf_text)  

        if duplicated_files:  # ตรวจสอบว่ามีไฟล์ที่ชื่อซ้ำหรือไม่
            flash(f"ไฟล์ชื่อซ้ำ ไม่ถูกบันทึก: {', '.join(duplicated_files)}", "warning")  

        return redirect(url_for('admin_upload')) 
    return render_template('admin_upload.html')  



# ----------------------------  
# Download Route
# ----------------------------
@app.route("/download/<int:thesis_id>")
def download_file(thesis_id):
    thesis = Thesis.query.get(thesis_id)
    if thesis:
        return send_from_directory(
            directory=UPLOAD_FOLDER,  # โฟลเดอร์เก็บไฟล์
            path=thesis.filename,     # ชื่อไฟล์
            as_attachment=True        # บังคับดาวน์โหลด
        )
    return "File not found", 404


# ---------------------------  
# หน้า login
# ---------------------------
@app.route('/')  
def index(): 
    return render_template('login.html')  

# ---------------------------  
# เริ่ม OAuth flow
# ---------------------------
@app.route('/login')  
def login():  
    google = OAuth2Session(GOOGLE_CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)  # สร้าง session สำหรับการขอสิทธิ์เข้าถึงข้อมูลผู้ใช้จาก Google
    authorization_url, state = google.authorization_url(AUTHORIZATION_BASE_URL,  # สร้าง URL สำหรับการอนุญาตจาก Google
                                                        access_type="offline",  # ขอสิทธิ์ offline access เพื่อรับ refresh token
                                                        prompt="select_account")  # บังคับให้ผู้ใช้เลือกบัญชีทุกครั้ง
    session['oauth_state'] = state  # เก็บค่า state ไว้ใน session เพื่อใช้ตรวจสอบความถูกต้องเมื่อ callback กลับมา
    return redirect(authorization_url)  # เปลี่ยนเส้นทางไปยังหน้าอนุญาตของ Google




# ---------------------------  
# Callback หลัง login
# ---------------------------
@app.route('/callback')  # เส้นทาง callback ที่ Google จะเรียกกลับหลังจากผู้ใช้อนุญาต
def callback():  # ฟังก์ชันสำหรับจัดการหลังจากผู้ใช้ล็อกอินผ่าน Google สำเร็จ
    google = OAuth2Session(GOOGLE_CLIENT_ID, state=session['oauth_state'], redirect_uri=REDIRECT_URI)  # สร้าง session ใหม่โดยใช้ state เดิมที่เก็บไว้
    token = google.fetch_token(TOKEN_URL, client_secret=GOOGLE_CLIENT_SECRET,  # ขอรับ token จาก Google
                               authorization_response=request.url)  # ส่ง URL ปัจจุบันเพื่อยืนยันการอนุญาต
    session['oauth_token'] = token  # เก็บ token ที่ได้รับไว้ใน session

    # ดึงข้อมูล user
    resp = google.get('https://www.googleapis.com/oauth2/v1/userinfo')  # เรียก API ของ Google เพื่อดึงข้อมูลผู้ใช้
    user_info = resp.json()  # แปลงข้อมูลที่ได้ให้อยู่ในรูปแบบ JSON
    
    # ตรวจสอบ email ว่าเป็นของมหาวิทยาลัยศิลปากร
    if user_info.get('email', '').endswith('@silpakorn.edu'):  # ตรวจสอบว่าอีเมลลงท้ายด้วย @silpakorn.edu หรือไม่
        session['user_name'] = user_info.get('name')  # เก็บชื่อผู้ใช้ไว้ใน session
        session['user_email'] = user_info.get('email')  # เก็บอีเมลผู้ใช้ไว้ใน session
        session['user_avatar_url'] = user_info.get('picture', '/static/default_avatar.png')  # เก็บ URL รูปโปรไฟล์ 
        return redirect(url_for('chat'))  
    else:
        return "บัญชีไม่ใช่อีเมลของมหาวิทยาลัยศิลปากร", 403  





# ----------------------------  
# Chatbot UI
# ----------------------------
@app.route('/chat')  
def chat():  
    if 'user_email' not in session:  
        return redirect(url_for('index'))  
    
    return render_template(  
        'chat.html',  
        user_name=session.get('user_name'),  
        user_email=session.get('user_email'), 
        user_avatar_url=session.get('user_avatar_url', '/static/default_avatar.png') 
    )


# ----------------------------  
# Chat API with Session Memory + RAG + Typhoon + Debug
# ----------------------------
@app.route('/api/chat', methods=['POST'])  
def api_chat():  
    question = request.json.get('question', '').strip()  # ดึงคำถามจาก JSON request และลบช่องว่างหัวท้าย
    if not question:  
        return jsonify({'reply': 'กรุณาส่งคำถามของผู้ใช้ด้วย'}), 400  

    # ----------------------------  
    # Session management
    # ----------------------------
    session_id = session.get('session_id')  # ตรวจสอบว่ามี session_id ใน session แล้วหรือยัง
    if not session_id:  # ถ้ายังไม่มี session_id
        session_id = str(uuid.uuid4())  # สร้าง session_id ใหม่แบบสุ่ม (UUID)
        session['session_id'] = session_id  # บันทึก session_id ลงใน session

    # ----------------------------  
    # Classify intent
    # ----------------------------
    intent = classify_intent(question)  
    reply = ""  

    # ----------------------------
    # Process intents
    # ----------------------------

# ----------------------------  
# Ranking
# ----------------------------
    if intent == "ranking": 
    # วิเคราะห์คำถาม
        year = extract_year(question)
        main_keyword, related_keywords = extract_keyword(question)

        if year and not main_keyword:
            related_keywords = []

        all_keywords = []
        if main_keyword:
            all_keywords = [main_keyword] + related_keywords

        log_debug(intent, {"year": year, "keyword": main_keyword, "related_keywords": related_keywords})

        results = []

    # 1️⃣ มีเฉพาะปี → query database ตาม year
        if year and not main_keyword:
            results = Thesis.query.filter_by(year=year).all()

    # 2️⃣ มี keywords แต่ไม่มี year → search FAISS
        elif main_keyword and not year:
            faiss_query = " ".join(all_keywords)
            doc_ids = search_Ranking(faiss_query, top_k_chunks=50, score_threshold=0.0)

            if doc_ids:
                results = Thesis.query.filter(Thesis.id.in_(doc_ids)).all()
                print("results",results)

    # 3️⃣ มีทั้ง year + keywords → search FAISS แล้ว filter ปี
        elif year and main_keyword:
            faiss_query = " ".join(all_keywords)
            print("faiss_query",faiss_query)
            doc_ids = search_Ranking(faiss_query, top_k_chunks=50, score_threshold=0.0)

            if doc_ids:
                results = Thesis.query.filter(
                    Thesis.id.in_(doc_ids),
                    Thesis.year == year
                ).all()

    # ตรวจสอบว่าพบข้อมูลหรือไม่
        if not results:
            if year and not main_keyword:
                reply = f"ไม่พบปริญญานิพนธ์ในปี พ.ศ. {year} กรุณาลองใหม่อีกครั้ง"
            elif main_keyword:
                reply = f"ไม่พบปริญญานิพนธ์เกี่ยวกับ \"{main_keyword}\" กรุณาลองใหม่อีกครั้ง"
            else:
                reply = "กรุณาระบุปีหรือคำค้นเพื่อค้นหารายชื่อปริญญานิพนธ์"
            return jsonify({"reply": reply})

    # สร้าง context
        context_lines = []
        for i, t in enumerate(results):
            context_lines.append( f"{i+1}. ชื่อเรื่อง: {t.title} | ปี: {t.year} | ผู้จัดทำ: {t.author} | อาจารย์ที่ปรึกษา: {t.advisor}")

        context = "\n".join(context_lines)
        print("สิ่งที่ส่งให้ Typhoon:", context)

    # สร้าง prompt ให้ Typhoon สรุปผล
        prompt = f"""  
คุณเป็นผู้ช่วยที่ช่วยเรียงลำดับและจัดหมวดหมู่ชื่อปริญญานิพนธ์
คำถามของผู้ใช้: "{question}"
นี่คือรายการปริญญานิพนธ์ที่เกี่ยวข้องจากฐานข้อมูล:
{context}
จัดเรียงคำตอบให้เข้าใจง่ายในรูปแบบภาษาไทยที่เป็นธรรมชาติ:
- จัดลำดับหรือจัดกลุ่มให้สวยงาม
- ต้องใช้ทุกปริญญานิพนธ์ที่ฉันส่งให้ห้ามตัดชื่อปริญญานิพนธ์ออกและห้ามสรุป
- ต้องบอกชื่อเรื่อง ปี ผู้จัดทำ อาจารย์ที่ปรึกษาด้วยตามลำดับ
- ใช้ภาษาสละสลวย เหมาะกับการตอบผู้ใช้
- ข้อความต่อท้ายให้บอกผู้ใช้ในลักษณะสามารถสอบถามข้อมูลเพิ่มเติมเกี่ยวกับปริญานิพนธ์ได้
"""

    # ส่ง prompt ไป Typhoon
        try:
            refined_reply = ask_typhoon(prompt).strip()  
            reply = refined_reply 
        except Exception as e: 
            print("⚠️ Typhoon error:", e)  
            reply = f"รายการปริญญานิพนธ์ที่พบ:\n{context} กรุณาลองใหม่อีกครั้ง"  

        return jsonify({"reply": reply})  
    

# ----------------------------  
# Download
# ----------------------------
    elif intent == "download": 
        title = extract_title(question)  
        log_debug(intent, {"title": title}) 

        thesis = Thesis.query.filter(Thesis.title.ilike(f"%{title}%")).first()  
        print("thesis",thesis) 
        if thesis:  
            dl_url = url_for('download_file', thesis_id=thesis.id)  
            print("dl_url",dl_url)
            reply = f'ดาวน์โหลดไฟล์ "{thesis.title}" ได้ที่นี่: <a href="{dl_url}">{thesis.title}</a>'
        else:  
            reply = f"ไม่พบชื่อปริญญานิพนธ์ที่ตรงกับ \"{title}\" กรุณาลองใหม่อีกครั้ง"





# ----------------------------  
# metadata_query
# ----------------------------
    elif intent == "metadata_query":  
        year = extract_year(question)  
        title = extract_title(question)  
        log_debug(intent, {"year": year, "title": title})  

        results = Thesis.query #คำสั่งsql
        if year: results = results.filter_by(year=year)  
        if title: results = results.filter(Thesis.title.ilike(f"%{title}%")) 
        results = results.all() #doc_id
        print("results",results)
        if not results:  
            reply = "ไม่พบข้อมูล metadata ที่ตรงกับคำถามกรุณาลองใหม่อีกครั้ง"  
        else: 
            reply = "\n".join([f"- {t.title}: ผู้ทำ {t.author}, อาจารย์ {t.advisor}, ปี {t.year}" for t in results])



# ----------------------------  
# answer_based_on_documents
# ----------------------------
    elif intent == "answer_based_on_documents": 
    # 🧠 ขั้นตอนที่ 1: ใช้ Typhoon สกัด keyword ที่หลากหลาย
        keyword_prompt = f"""  
    จากคำถามนี้ "{question}" 
    จงสกัด:
    1. คำหลักของหัวข้อวิทยานิพนธ์ (title keywords)
    2. ประเด็นหรือหัวข้อเฉพาะ (topic keywords)
    3. ถ้ามีการอ้างถึงบท เช่น บทที่ 1 หรือ บทที่ 3 ให้ระบุออกมาเป็น "chapter"
    ตอบในรูปแบบ JSON เช่น:
    {{"title_keywords": ["..."], "topic_keywords": ["..."], "chapter": "บทที่ 3"}}
    """
        print(f"[DEBUG] Keyword extraction prompt:\n{keyword_prompt}\n{'-'*50}")  
        keyword_json = ask_typhoon(system_prompt=keyword_prompt).strip() 

        # ลบ ```json และ ``` รอบ ๆ JSON (ถ้ามี)
        keyword_json_clean = keyword_json
        if keyword_json_clean.startswith("```json"):
            keyword_json_clean = keyword_json_clean[len("```json"):].strip()
        if keyword_json_clean.startswith("```"):  # กรณีไม่มี json แต่มี backticks
            keyword_json_clean = keyword_json_clean[3:].strip()
        if keyword_json_clean.endswith("```"):
            keyword_json_clean = keyword_json_clean[:-3].strip()

        try: 
            keyword_data = json.loads(keyword_json_clean)  # แปลง JSON เป็น dict
            title_keywords = ", ".join(keyword_json_clean.get("title_keywords", []))  # รวม title_keywords เป็น string
            topic_keywords = ", ".join(keyword_json_clean.get("topic_keywords", []))  # รวม topic_keywords เป็น string
            chapter = keyword_data.get("chapter", "")  # ดึงค่า chapter
        except:  
            title_keywords = question  # ใช้คำถามทั้งหมดเป็น title_keywords
            topic_keywords = ""  
            chapter = ""  

        search_query = " ".join(filter(None, [title_keywords, topic_keywords, chapter]))  # รวมคำค้นทั้งหมดเป็น query สำหรับค้นหาเอกสาร

    # 🔍 ขั้นตอนที่ 2: ค้นหาเอกสารจาก keyword ที่สกัดได้
        docs, best_doc_id = search_similar(search_query, max_chunks=50)  
        log_debug(intent, {"best_doc_id": best_doc_id, "num_docs": len(docs)})  
        
        if not docs or len(docs) == 0:  
            reply = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองถามใหม่อีกครั้ง"  
            return reply 
        
    # 🧩 ขั้นตอนที่ 3: รวมเนื้อหาจาก chunks ที่ได้
        print("docs",docs)
        context = "\n".join([c['text'] for c in docs])  # รวมเนื้อหาของแต่ละ chunk เป็นข้อความเดียว

    # 💬 ขั้นตอนที่ 4: เตรียม prompt สำหรับ LLM ให้เข้าใจว่าต้องตอบ “เฉพาะส่วนของเอกสาร”
        full_prompt = f"""  
    คุณเป็นผู้ช่วยที่ใช้ข้อมูลจากวิทยานิพนธ์เพื่อตอบคำถามอย่างถูกต้องและตรงประเด็น

    บริบทจากเอกสาร:
    {context}

    คำถามของผู้ใช้: "{question}"
    ผู้ใช้ระบุบท: {chapter or "ไม่ได้ระบุบท"}

    หากมีการระบุบท ให้สรุปและอธิบายเนื้อหาเฉพาะในบทนั้นเท่านั้น
    ห้ามอ้างอิงเนื้อหาจากบทอื่น ห้ามสรุปเนื้อหานอกบริบทที่กำหนด
    หากไม่มีการระบุบท ให้ตอบจากภาพรวมของบริบททั้งหมด

    ตอบอย่างละเอียด ถูกต้อง และอิงข้อมูลจากเอกสารเท่านั้น
    """

        reply = ask_typhoon(system_prompt=full_prompt)  

    # 🗂️ ขั้นตอนที่ 5: ดึง metadata ของเอกสาร
        doc_meta = None  
        if best_doc_id:  
            best_doc = Thesis.query.filter_by(id=int(best_doc_id)).first()  
            if best_doc: 
                doc_meta = {  # สร้าง dictionary เก็บ metadata
                    "title": best_doc.title,
                    "author": best_doc.author,
                    "advisor": best_doc.advisor,
                    "year": best_doc.year
            }

    # 🧾 ขั้นตอนที่ 6: แสดง metadata ต่อท้ายคำตอบ
        if doc_meta: 
            metadata_text = f"""  
            สามารถขอโหลดเอกสารเพื่อดูข้อมูลเพิ่มเติมได้
            📘 ข้อมูลวิทยานิพนธ์ที่เกี่ยวข้อง:
            - ชื่อเรื่อง: {doc_meta['title']}
            - ผู้จัดทำ: {doc_meta['author']}
            - อาจารย์ที่ปรึกษา: {doc_meta['advisor']}
            - ปีที่จัดทำ: {doc_meta['year']}
            """
            reply += metadata_text  


    
# ----------------------------  
# lecturer_info
# ----------------------------
    elif intent == "lecturer_info":  
        log_debug(intent, {"question": question})  
        try: 
            keyword = ask_typhoon(
                system_prompt=f"""  
คุณได้รับข้อความจากผู้ใช้: '{question}'
โปรดดึงคำสำคัญออกมาเฉพาะ 1-3 คำที่เกี่ยวข้องกับ:
- ชื่ออาจารย์
- สาขาวิชา
- ความถนัด/ความเชี่ยวชาญ

ข้อกำหนด:
1. ไม่ต้องส่งสัญลักษณ์ใด ๆ เช่น . , ; : หรือเครื่องหมายพิเศษ
2. ไม่ต้องส่งคำที่ไม่เกี่ยวข้อง
3. แยกคำสำคัญแต่ละคำด้วยเครื่องหมายจุลภาค (,)
4. ส่งผลลัพธ์เป็นข้อความสั้น ๆ ที่สามารถนำไปใช้ search ในฐานข้อมูลได้ทันที
5. ตัดคำออกเช่น อาจารย์ , ภาค , สาขา , วิชา เป็นต้น
ตัวอย่างผลลัพธ์ที่ถูกต้อง: "สมชาย, คอมพิวเตอร์, Computer Network Architectures"
"""
        ).strip()  
        except:  
            keyword = question 

        keywords = [k.strip() for k in re.split(r"[,;]", keyword) if k.strip()]  # แยก keyword เป็นลิสต์โดยใช้ , หรือ ; เป็นตัวแบ่ง และลบช่องว่าง
        query = Lecturer.query #คำสั่งsql
        print("query",query)
        print("keyword",keyword)
        if keywords: 
            filters = [] 
            for k in keywords:  
                print("ค่าk",k)
                filters.append(Lecturer.name.ilike(f"%{k}%")) 
                filters.append(Lecturer.department.ilike(f"%{k}%"))  
                filters.append(Lecturer.expertise.ilike(f"%{k}%")) 
            query = query.filter(or_(*filters))  # รวมเงื่อนไขทั้งหมดด้วย OR
        print("query",query)
        results = query.all() #ชื่ออาจารย์ทั้งหมด
        print("results",results)
        log_debug(intent, {"num_results": len(results)})  

        def format_lecturer(lec):  # ฟังก์ชันจัดรูปแบบข้อความแสดงอาจารย์
            text = f"- {lec.title + ' ' if hasattr(lec, 'title') and lec.title else ''}{lec.name}: สาขา {lec.department}, เชี่ยวชาญ {lec.expertise}" 
            if lec.link:  
                text += f'<br>🔗 <a href="{lec.link}" target="_blank">ดูข้อมูลเพิ่มเติม</a>' 

            return text  

        if not results:  
            reply = f"ไม่พบอาจารย์ที่เกี่ยวข้องกับ '{keyword}' กรุณาลองใหม่อีกครั้ง" 

        elif len(results) == 1:  
            lec = results[0]
            reply = format_lecturer(lec) 

        else:  
            context_text = "\n".join([format_lecturer(lec) for lec in results])  
            prompt_reply = f"""คุณได้รับคำถามจากผู้ใช้: "{question}"
นี่คือรายชื่ออาจารย์ที่ตรงเบื้องต้น:
{context_text}
โปรดตอบกลับเฉพาะอาจารย์ที่ตรงกับคำถามผู้ใช้ ตอบเป็นข้อความอ่านง่าย"""  
            try:  
                reply = ask_typhoon(system_prompt=prompt_reply).strip() 
            except: 
                reply = "อาจารย์ที่ตรงกับคำถามของคุณ:\n" + context_text  


# ----------------------------  
# project_advice
# ----------------------------
    elif intent == "project_advice":  
        log_debug(intent, {"question": question})  
        prompt = f"ผู้ใช้ถาม: '{question}'\nคุณเป็นผู้ช่วยให้คำปรึกษาโครงงาน ให้คำแนะนำไอเดียโปรเจคที่เหมาะสม อ่านง่าย และสร้างสรรค์โดยบอกด้วยว่าTyphoonเป็นคนให้คำแนะนำไม่ได้เอามาจากฐานข้อมูลภาคคอมพิวเตอร์"  
        try:  
            reply = ask_typhoon(system_prompt=prompt) 
        except: 
            reply = "ขอโทษ ฉันไม่สามารถให้คำแนะนำโปรเจคได้ตอนนี้ ลองถามใหม่"  


# ----------------------------  
# advisor_projects
# ----------------------------
    elif intent == "advisor_projects":  
        log_debug(intent, {"question": question})  

        try:
            keyword = ask_typhoon(
                system_prompt=f"จากข้อความนี้ '{question}' ดึงเฉพาะชื่อ ห้ามมีคำว่า อาจารย์ ดร. ผศ. รศ. หรือคำนำหน้าอื่น ๆ"
            ).strip() 
        except:  
            keyword = question  
            
        keywords = [keyword] 
        print("keywords",keywords)
   
        query = Thesis.query  #คำสั่งsql
        if keywords: 
            filters = [Thesis.advisor.ilike(f"%{k}%") for k in keywords]  #ค้นหาอาจารย์ที่ตรงในThesis
            query = query.filter(or_(*filters))  #คำสั่งsql
            print("query",query) 

        results = query.all() #ชื่อไฟล์ที่คิวรี่มา
        print("results",results)
        log_debug(intent, {"num_results": len(results)})  

        if not results: 
            reply = f"ไม่พบโปรเจคที่อาจารย์ '{keyword}' เป็นที่ปรึกษา กรุณาลองใหม่อีกครั้ง"  
        else: 
        
            context_text = "\n".join([f"- {t.title}, ผู้ทำ: {t.author}, ปี: {t.year}" for t in results]) 
            prompt_reply = f"""  
คุณได้รับคำถามจากผู้ใช้: "{question}"
นี่คือรายการโปรเจคที่อาจารย์ '{keyword}' เป็นที่ปรึกษา:
{context_text}

โปรดตอบกลับผู้ใช้เป็นข้อความที่อ่านง่าย เป็นมิตร และเข้าใจง่าย
จัดเรียงข้อมูลให้สวยงาม เช่น แยกเป็นหัวข้อหรือประโยค ไม่ต้องใช้เครื่องหมาย bullet เยอะ
"""
            try:  
                reply = ask_typhoon(system_prompt=prompt_reply).strip()  
            except:  
                reply = "โปรเจคที่อาจารย์นี้เป็นที่ปรึกษา:\n" + context_text  


# ----------------------------  
# chitchat
# ----------------------------
    elif intent == "chitchat" or intent == "unknown": 
        log_debug(intent, {"question": question})  
        prompt = f"ผู้ใช้ถาม: '{question}'\nตอบเป็นข้อความของบอทอย่างเป็นมิตร อ่านง่าย"  
        print(f"[DEBUG] Prompt sent to LLM (chitchat/unknown):\n{prompt}\n{'-'*50}") 
        try:  
            reply = ask_typhoon(system_prompt=prompt)  
        except:  
            reply = "ขอโทษ ฉันมีปัญหาในการตอบคำถามตอนนี้ ลองถามใหม่ได้ไหม?" 

    return jsonify({'reply': reply}) 


# ---------------------------
# How-to-use
# ---------------------------

@app.route('/how-to-use')  
def how_to_use():  
    if 'user_email' not in session: 
        return redirect(url_for('index')) 
    return render_template('how_to_use.html', name=session.get('user_name'))


# ---------------------------
# Logout
# ---------------------------
@app.route('/logout', methods=['GET', 'POST'])  
def logout(): 
    session.clear()  
    return redirect(url_for('index')) 



# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':  # ตรวจสอบว่าไฟล์นี้ถูกเรียกใช้โดยตรง ไม่ใช่ import เป็นโมดูล
    if not os.path.exists(UPLOAD_FOLDER):  # ตรวจสอบว่าโฟลเดอร์สำหรับเก็บไฟล์ PDF มีอยู่หรือไม่
        os.makedirs(UPLOAD_FOLDER)  # ถ้าไม่มี ให้สร้างโฟลเดอร์
    app.run(debug=True)  # รัน Flask app ในโหมด debug (เพื่อให้เห็น error และ reload อัตโนมัติ)

