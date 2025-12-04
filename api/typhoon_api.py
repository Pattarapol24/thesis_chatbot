import requests  
from config import TYPHOON_API_KEY  

def ask_typhoon(system_prompt):
    url = "https://api.opentyphoon.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TYPHOON_API_KEY}",  
        "Content-Type": "application/json"  
    }
    data = {
        "model": "typhoon-v2.1-12b-instruct",  
        "messages": [
            {"role": "system", "content": "นายมีชื่อว่า ThesisChatbot ที่จะตอบคำถามเกี่ยวกับปริญานิพนธ์ ของภาควิชาคอมพิวเตอร์ มหาวิทยาลัยศิลปากร "
            "สิ่งที่นายสามารถทำได้คือ ช่วยหาข้อมูลปริญานิพนธ์ ของภาควิชาคอมพิวเตอร์ มหาวิทยาลัยศิลปากรได้อย่างเดียว โดยจะไม่คิดและตอบเองถ้าฉันไม่ได้บอก "
            "คุณต้องไม่ตอบคำถามเกี่ยวกับเนื้อหาที่ไม่เหมาะสม, ความรุนแรง, ลามก, การพนัน หรือการละเมิดกฎหมาย"},
            {"role": "user", "content": system_prompt}  
        ],
        "max_tokens": 5000,  
        "temperature": 0.1 
    }
    try:
        res = requests.post(url, headers=headers, json=data, timeout=600)
        res.raise_for_status() 
        r = res.json()
        reply = r["choices"][0]["message"]["content"].strip()
        print(f"[Typhoon reply] {reply}")
        return reply  
    except Exception as e:
        print(f"[Typhoon API ERROR] {e}")
        return f"เกิดข้อผิดพลาดจาก API: {e}"  
