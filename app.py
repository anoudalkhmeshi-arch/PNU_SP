import streamlit as st
import requests
import random
import base64
import os

# 1. PAGE CONFIG & RTL STYLING

st.set_page_config(page_title="PNU SmartPark", page_icon="🅿️", layout="wide")
API_URL = "http://127.0.0.1:8000/api"

def get_local_img(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None
current_folder = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(current_folder, "pnu-logo-ar-h.svg")
img_base64 = get_local_img(LOGO_PATH)

st.markdown("""
<style>
    /* Full RTL Support */
    .stApp { direction: rtl; font-family: 'Arial', sans-serif; text-align: right; }
    [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    p, h1, h2, h3, h4, h5, h6, label { text-align: right; }
    .stButton > button { width: 100%; font-size: 11px; font-weight: bold; margin-bottom: 5px; }
    
    /* تنسيق اللوقو */
    .logo-box {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 5px;
        background-color: transparent; 
        margin-bottom: 20px;
        width: 100%;
        height: 120px;
    }
    .logo-box img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    
    /* Navigation Bar Style */
    .nav-container { display: flex; justify-content: center; gap: 15px; padding: 15px; background: #f8f9fa; border-radius: 12px; margin-bottom: 30px; border: 1px solid #ddd; }
    .nav-link { text-decoration: none; color: #2E86C1; font-weight: bold; padding: 8px 18px; border: 2px solid #2E86C1; border-radius: 8px; transition: 0.3s; }
    .nav-link:hover { background: #2E86C1; color: white; }
</style>
""", unsafe_allow_html=True)

STATIONS = ["A1", "A2", "A3", "A4", "A5", "A6"]

# Sidebar on the right
with st.sidebar:
    if img_base64:
        st.markdown(f"""
            <div class="logo-box">
                <img src="data:image/svg+xml;base64,{img_base64}">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("لم يتم العثور على اللوقو في المسار المحدد")
        
    st.title("بوابة المواقف")
    page = st.radio("القائمة الرئيسية", ["المراقبة والحجز", "محاكاة الكاميرا YOLO"])

def get_data():
    try: return requests.get(f"{API_URL}/status").json()["data"]
    except: return []


# PAGE 1: MONITORING & RESERVATION

if page == "المراقبة والحجز":
    st.markdown("<h1 style='text-align:center; color:#2E86C1;'>PNU SmartPark🅿️</h1>", unsafe_allow_html=True)
    
    # 1. QUICK NAVIGATION BAR (Jump to Station)
    nav_html = '<div class="nav-container">'
    for s in STATIONS:
        nav_html += f'<a class="nav-link" href="#{s.lower()}">محطة {s}</a>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)
    
    data = get_data()
    if not data:
        st.error("خطأ: تعذر الاتصال بالخادم. (Backend Offline)")
    else:
        for lot in data:
            s_id = lot['station_id']
            # Create Anchor for Navigation
            st.markdown(f"<div id='{s_id.lower()}'></div>", unsafe_allow_html=True)
            
            with st.container(border=True):
                st.subheader(f"📍 المحطة {s_id}")
                
                # Metrics and Forecasting
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 إجمالي المشغول", lot['occupied_spots'])
                c2.metric("🟢 المتاح حالياً", lot['free_spots'])
                
                f_res = requests.get(f"{API_URL}/forecast/{s_id}").json()
                c3.metric("توقع للازدحام للساعةالقادمة", f"{f_res.get('prediction', 0) * 100:.1f}%")
                
                st.divider()
                st.write("**خريطة المواقف (🟩: متاح | 🟥: مشغول):**")
                
                # GRID LOGIC (Cumulative)
                total = lot['total_capacity']
                res_indices = lot['reserved_indices']
                yolo_needed = max(0, lot['occupied_spots'] - len(res_indices))
                
                # Distribute YOLO detections randomly among non-reserved spots
                random.seed(s_id)
                available_indices = [i for i in range(total) if i not in res_indices]
                yolo_indices = random.sample(available_indices, min(len(available_indices), yolo_needed))
                all_red = set(res_indices).union(set(yolo_indices))

                # Draw Grid (10 columns)
                cols = st.columns(10)
                for i in range(total):
                    idx = i % 10
                    label = f"{s_id}-{i+1:02d}"
                    if i in all_red:
                        cols[idx].button(f"🟥 {label}", key=f"btn_{s_id}_{i}", disabled=True)
                    else:
                        if cols[idx].button(f"🟩 {label}", key=f"btn_{s_id}_{i}"):
                            st.session_state.active_reserve = {"station_id": s_id, "index": i, "label": label}

                # Reservation Form (Mandatory Fields)
                if "active_reserve" in st.session_state and st.session_state.active_reserve['station_id'] == s_id:
                    with st.form(f"form_{s_id}"):
                        st.info(f"حجز الموقف رقم: {st.session_state.active_reserve['label']}")
                        
                        student_id = st.text_input("الرقم الجامعي (مطلوب لإتمام الحجز):")
                        t1, t2 = st.columns(2)
                        with t1: s_time = st.time_input("بداية الحجز (From):")
                        with t2: e_time = st.time_input("نهاية الحجز (To):")
                        
                        if st.form_submit_button("تأكيد الحجز النهائي ✅"):
                            if not student_id.strip():
                                st.error("❌ لا يمكن الحجز بدون إدخال الرقم الجامعي!")
                            elif s_time >= e_time:
                                st.error("❌ خطأ في الوقت: يجب أن تكون النهاية بعد البداية.")
                            else:
                             
                                response = requests.post(f"{API_URL}/reserve", json={
                                    "student_id": student_id, "station_id": s_id, 
                                    "spot_index": st.session_state.active_reserve['index'],
                                    "start_time": str(s_time), "end_time": str(e_time)
                                })
                                
                             
                                if response.status_code == 200:
                                 
                                    del st.session_state.active_reserve
                                    st.success("تم حجز الموقف بنجاح! 🟩")
                                    st.rerun()
                                    
                                elif response.status_code == 400:
                                
                                    error_msg = response.json().get("detail", "")
                                    if "Conflict" in error_msg:
                                        st.error("❌ عذراً، هذا الموقف محجوز مسبقاً من شخص آخر في نفس الوقت! الرجاء اختيار موقف أو وقت آخر.")
                                    elif "Capacity" in error_msg:
                                        st.error("🚫 عذراً، المحطة ممتلئة بالكامل حالياً!")
                                    else:
                                        st.error("❌ حدث خطأ أثناء الحجز، حاولي مرة أخرى.")
                                        
                                else:
                                    st.error("❌ تعذر الاتصال بالخادم (Backend Offline).")
                    
                st.markdown("---")

# PAGE 2: YOLO CAMERA SIMULATION

else:
    st.title("محاكاة الكاميرا (YOLO) 📷")
    station_sel = st.selectbox("اختر المحطة المرتبطة بالكاميرا:", STATIONS)
    file = st.file_uploader("ارفع صورة المحطة المباشرة", type=["jpg", "png", "jpeg"])
    
    if file:
        st.image(file, caption="تحليل الكاميرا المباشر", use_container_width=True)
        if st.button("بدء الفحص وتحديث البيانات التراكمية"):
            with st.spinner("جاري استخراج السيارات..."):
                res = requests.post(f"{API_URL}/detect/{station_sel}", files={"file": (file.name, file.getvalue())}).json()
                if res["status"] == "success":
                    st.success(f"✅ تم الرصد! الكاميرا كشفت {res['new_yolo_count']} سيارات في المحطة {station_sel}.")
                    st.info("تم دمج الرصد مع الحجوزات النشطة في الصفحة الرئيسية.")
