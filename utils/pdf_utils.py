import pdfplumber  
import re  
#\bขอบเขตคำ \sช่องว่าง \dตัวเลข \nขึ้นบรรทัดใหม่ $จุดสิ้นสุดบรรทัด

def clean_text(text):  # ฟังก์ชันลบช่องว่างเกินและทำความสะอาดข้อความ
    text = text.replace('\n', ' ')  # แทนที่ new line ด้วยช่องว่าง replace = แทนที่ตรงๆ
    text = re.sub(r'\s+', ' ', text)  # ลบช่องว่างซ้ำ ๆ ให้เหลือช่องว่างเดียว re.sub = ตามรูปแบบ
    text = text.strip()  # ลบช่องว่างที่หน้า/หลังข้อความ
    text = text.replace('.', '')  # ลบจุด
    text = text.replace('…', '')  # ลบจุดสามจุด
    return text  

def remove_page_number(text):  # ฟังก์ชันลบเลขหน้าที่อยู่ในข้อความ
    text = re.sub(r'\b(หน้า|Page|page)\s*\d+\b', '', text)  # ลบคำว่า หน้า/Page ตามด้วยตัวเลข
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # ลบตัวเลขบรรทัดเดียวทั้งหมด
    text = re.sub(r'(^|\n)\s*\d+\s*(\n|$)', ' ', text)  # ลบตัวเลขที่ขึ้นบรรทัดใหม่
    return text 

def fix_thai_vowel_typos(text):  # ฟังก์ชันแก้ไขสระหรือวรรณยุกต์ภาษาไทยที่เพี้ยนจาก Unicode PUA
    typo_map = {  # แผนที่อักขระผิด : อักขระถูกต้อง
        '\uF702': 'ี',   
        '\uF70A': '่',
        '\uF70B': '้',
        '\uF70C': '๊',
        '\uF70D': '๋',
        '\uF70E': '์',
        '\uF70F': 'ํ',
        '\uF710': 'ุ',
        '\uF711': 'ู',
        '\uF712': 'ิ',
        '\uF713': 'ี',
        '\uF714': 'ึ',
        '\uF715': 'ื',
        '\uF716': 'ั',
        '\uF717': 'า',
        '\uF718': 'เ',
        '\uF719': 'แ',
        '\uF71A': 'โ',
        '\uF71B': 'ใ',
        '\uF71C': 'ไ',
    }
    for wrong, correct in typo_map.items():  # แทนที่ตัวอักษรผิดทั้งหมดด้วยตัวถูกต้อง .itemsเมธอดpythonดึงคู่key:value
        text = text.replace(wrong, correct)
    return text  

def fix_split_vowels(text):  # ฟังก์ชันรวมสระ/วรรณยุกต์ที่แยกด้วยช่องว่างกับพยัญชนะ
    """
    รวมพยัญชนะกับวรรณยุกต์/สระที่ถูกแยกด้วยช่องว่าง
    เช่น "คอมพิวเตอร ์" -> "คอมพิวเตอร์", "การท า" -> "การทำ"
    """
    pattern = re.compile(r'([ก-ฮ])\s+([่้๊๋์ํุูิีึืัาเแโใไ])')  # re.compile สร้าง pattern objec: พยัญชนะ + ช่องว่าง + สระ/วรรณยุกต์
    return pattern.sub(r'\1\2', text)  # แทนที่และส่งกลับ

def extract_text_from_pdf(pdf_path):  # ฟังก์ชันอ่านและประมวลผลข้อความจาก PDF
    text = ""  
    with pdfplumber.open(pdf_path) as pdf:  # เปิดไฟล์ PDF
        for page in pdf.pages:  # วนลูปทุกหน้า
            page_text = page.extract_text() or ""  # ดึงข้อความหน้า ถ้าไม่มีให้เป็น ""
            page_text = remove_page_number(page_text)  # ลบเลขหน้า
            text += page_text  

    text = fix_thai_vowel_typos(text)   
    text = fix_split_vowels(text)       
    text = clean_text(text)           
    return text  



