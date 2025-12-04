from flask import Flask, render_template, request, jsonify  
import sqlite3  

app = Flask(__name__)  
db_path = 'instance/db.sqlite3'  

# ------------------ หน้าแสดงตาราง ------------------
@app.route('/')  
def show_table(): 
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()  # สร้าง cursor
        cursor.execute("PRAGMA table_info(thesis);")  # ดึงข้อมูลคอลัมน์ของตาราง thesis
        columns = [col[1] for col in cursor.fetchall()]  # เก็บชื่อคอลัมน์
        cursor.execute("SELECT * FROM thesis;")  # ดึงข้อมูลทั้งหมดจากตาราง thesis
        rows = cursor.fetchall()  # เก็บผลลัพธ์
    return render_template('admin_doc.html', cols=columns, rows=rows) 

# ------------------ หน้าแก้ไขข้อมูล ------------------
@app.route('/admin/edit/<int:thesis_id>', methods=['GET', 'POST']) 
def admin_edit(thesis_id):  
    with sqlite3.connect(db_path) as conn:  # เชื่อมต่อฐานข้อมูล
        cursor = conn.cursor()  # สร้าง cursor = ตัวจัดการคำสั่ง SQL
        if request.method == 'GET':  # กรณีเปิดหน้าแก้ไข
            cursor.execute("SELECT id, title, author, advisor, year, filename FROM thesis WHERE id=?", (thesis_id,))  # ดึงข้อมูล thesis ตาม id
            row = cursor.fetchone()  # ดึงข้อมูลแถวเดียว
            if not row:  
                return "ไม่พบข้อมูล thesis", 404  
            thesis = {  # เตรียม dictionary สำหรับส่งไป template
                "id": row[0],
                "title": row[1],
                "author": row[2],
                "advisor": row[3],
                "year": row[4],
                "filename": row[5]
            }
            return render_template("admin_edit.html", thesis=thesis) 

        elif request.method == 'POST': 
            data = request.get_json()  # รับข้อมูล JSON จาก request
            title = data.get('title', '').strip()  
            author = data.get('author', '').strip() 
            advisor = data.get('advisor', '').strip()
            year = data.get('year', '').strip()  
            filename = data.get('filename', '').strip()  

            if not title or not author or not advisor or not year:  # ตรวจสอบข้อมูลครบหรือไม่
                return jsonify({"status":"error", "message":"กรุณากรอกข้อมูลให้ครบทุกช่อง"})  

            try:  
                cursor.execute("""  
                    UPDATE thesis
                    SET title=?, author=?, advisor=?, year=?, filename=?
                    WHERE id=?
                """, (title, author, advisor, year, filename, thesis_id))  # ใส่ค่า parameters
                conn.commit()  # commit การเปลี่ยนแปลง
                return jsonify({"status":"success"})  
            except Exception as e:  
                return jsonify({"status":"error", "message": str(e)}) 

if __name__ == '__main__': 
    app.run(debug=True, port=5001) 
