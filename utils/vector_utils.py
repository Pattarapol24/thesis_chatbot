import os  # นำเข้าโมดูลสำหรับจัดการไฟล์/โฟลเดอร์
import faiss 
import numpy as np  # นำเข้า NumPy สำหรับจัดการ array
from sentence_transformers import SentenceTransformer  
import pickle  # สำหรับบันทึก/โหลดไฟล์ Python object
from pythainlp.tokenize import word_tokenize  # สำหรับตัดคำภาษาไทย
import logging  
from collections import Counter  #Counterนับจำนวน ของสิ่งที่ซ้ำ ๆ ใน list 
# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')  # ตั้งค่า logging

# ----------------------------
# Path สำหรับเก็บฐานข้อมูล FAISS และ id_map
# ----------------------------
VECTOR_DB = "instance/faiss_index"  

# ----------------------------
# โหลดโมเดล embedding
# ----------------------------
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
EMBEDDING_SIZE = 384  

# ----------------------------
# โหลด FAISS และ id_map ถ้ามีอยู่แล้ว
# ----------------------------
if os.path.exists(VECTOR_DB):  
    with open(VECTOR_DB, 'rb') as f:  # เปิดไฟล์ FAISS index
        faiss_db, id_map = pickle.load(f)  # โหลด faiss_db และ id_map
    logging.info(f"โหลด FAISS index จาก {VECTOR_DB} สำเร็จ, มี {faiss_db.ntotal} vectors") 
else: 
    faiss_db = faiss.IndexFlatL2(EMBEDDING_SIZE)  # สร้าง FAISS index ใหม่
    id_map = {}  # สร้าง dictionary เปล่าสำหรับเก็บ mapping
    logging.info("สร้าง FAISS index ใหม่")  

# ----------------------------
# เพิ่มเอกสารลง FAISS
# ----------------------------
def add_doc_to_vectorstore(doc_id, text, chunk_size=2000): 
    global faiss_db, id_map  # ใช้ตัวแปร global = ฟังชั่นอื่นสามารถใช้ได้
    # แบ่ง paragraph (fallback ถ้าไม่มี \n\n)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]  # แบ่ง paragraph
    if not paragraphs: 
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]  # แบ่งตามบรรทัด

    # สร้าง chunks
    chunks = []  
    for para in paragraphs:  # วนลูป paragraph
        sentences = word_tokenize(para, engine="attacut")  # ตัดคำ/ประโยค
        temp_chunk = ""  # เตรียมตัวแปรเก็บ chunk ชั่วคราว
        for s in sentences:  # วนลูปแต่ละ sentence
            if len(temp_chunk) + len(s) > chunk_size:  # ถ้าเกิน chunk_size
                if temp_chunk:
                    chunks.append(temp_chunk)  # เก็บ chunk ก่อนหน้า append methodของ list Pythonใช้ เพิ่มสมาชิกเข้าไปใน listท้ายสุด
                temp_chunk = s  # เริ่ม chunk ใหม่
            else:
                temp_chunk = temp_chunk + " " + s if temp_chunk else s  # ต่อข้อความ
        if temp_chunk:
            chunks.append(temp_chunk)  # เก็บ chunk สุดท้าย 

    # แปลงหลาย chunk เป็น embedding vector พร้อมกัน
    vectors = model.encode(chunks, batch_size=32)  # encode เป็นเวกเตอร์
    for c, v in zip(chunks, vectors):  # วนลูป chunks และ vectors zipจับคู่
        faiss_db.add(np.array([v], dtype='float32'))  # เพิ่ม vector ลง FAISS
        idx = faiss_db.ntotal - 1  # ดัชนีล่าสุด -1เพราะindexเริ่ม0 ntotalคือเวกเตอร์ทั้งหมดที่เก็บไว้
        id_map[idx] = {'id': doc_id, 'text': c}  # บันทึก mapping

    # บันทึก FAISS + id_map
    os.makedirs(os.path.dirname(VECTOR_DB), exist_ok=True)  # สร้างโฟลเดอร์ถ้าไม่มี
    with open(VECTOR_DB, 'wb') as f:  # เปิดไฟล์สำหรับบันทึก
        pickle.dump((faiss_db, id_map), f)  # บันทึก FAISS + id_map

    logging.info(f"เพิ่มเอกสาร {doc_id} จำนวน {len(chunks)} chunks ลง FAISS เรียบร้อย")  

# ----------------------------
# ค้นหาเอกสารที่คล้ายกับ query
# ----------------------------
def search_similar(query, max_chunks=50):  
    if faiss_db.ntotal == 0:  # ถ้า FAISS ว่าง
        logging.warning("FAISS index ว่าง, ยังไม่มีเอกสาร")  
        return [], []  

    # encode query
    q_vec = np.array([model.encode([query])[0]], dtype='float32')  # แปลงคำถามผู้ใช้
    print("debug = ", q_vec)  
    # ค้นหา FAISS
    D, I = faiss_db.search(q_vec, max_chunks * 2)  # ค้นหาchunks ที่ใกล้เคียงมากที่สุดและคืน I ใช้ดึงข้อความจาก id_map และ D เป็น score

    # ดึง chunks พร้อม score
    found_chunks = []  # list เก็บผลลัพธ์
    for idx, score in zip(I[0], D[0]):  # วนลูป index และ score
        if idx in id_map:
            chunk = id_map[idx].copy()  # copy chunk
            chunk['score'] = score  # เพิ่ม score
            found_chunks.append(chunk)  # เก็บ chunk

    if not found_chunks:  
        return [], []  

    doc_id_counts = Counter(chunk['id'] for chunk in found_chunks)  # นับจำนวน chunk ต่อ doc_id
    best_doc_id = doc_id_counts.most_common(1)[0][0]  # เลือก doc_id ที่มี chunk มากที่สุด
    filtered_chunks = [c for c in id_map.values() if c['id'] == best_doc_id]  # filter **ทุก chunk** จาก doc_id นั้น (ไม่ตัด max_chunks)

    for c in filtered_chunks:  # วนลูป filtered_chunks
        c['score'] = next((f['score'] for f in found_chunks if f['text'] == c['text']), 0)  # เอาคะแนน similarity ของ chunk จาก found_chunks มาใส่ให้ c ถ้าไม่มี ให้เป็น 0
    filtered_chunks.sort(key=lambda x: x['score'])      # sort ตาม similarity score ของ chunk ที่ค้นเจอจาก FAISS

    logging.info(f"Search query: '{query}' -> เลือก doc_id {best_doc_id}, chunks {len(filtered_chunks)}")  
    return filtered_chunks, best_doc_id  # คืนค่า filtered chunks และ best_doc_id

def search_Ranking(query, top_k_chunks=50, score_threshold=0.0):
    """
    คืน doc_id ทั้งหมดที่มี chunk คล้าย query
    - top_k_chunks: จำนวน chunk สูงสุดที่จะ search ใน FAISS
    - score_threshold: กรอง chunk ที่ score ต่ำกว่านี้
    """
    print("faiss_db",faiss_db)
    if faiss_db.ntotal == 0:
        logging.warning("FAISS index ว่าง, ยังไม่มีเอกสาร")
        return []

    # encode query เป็น vector
    q_vec = np.array([model.encode([query])[0]], dtype='float32')

    # search FAISS เฉพาะ top_k_chunks
    D, I = faiss_db.search(q_vec, min(top_k_chunks, faiss_db.ntotal))#ค้นหาvectorที่ใกล้และทำการป้องกันจำนวนเกิน vector ที่มี

    doc_ids = set() #สร้าง set ว่าง สำหรับเก็บ doc_id
    for idx, score in zip(I[0], D[0]): #จับคู่ index + score ทีละตัว
        if idx in id_map and score >= score_threshold: #ถ้าความคล้ายมากกว่า0.0
            doc_ids.add(id_map[idx]['id']) #เพิ่มลงdoc_ids
        print(score)
    return list(doc_ids)
