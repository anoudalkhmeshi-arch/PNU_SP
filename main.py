from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import sqlite3
import pytz
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import random
import datetime as datetime_
from fastapi import HTTPException


        

class ReserveRequest(BaseModel):
    student_id: str
    station_id: str
    spot_index: int
    start_time: str
    end_time: str


# 1. SETUP APP & STATION CAPACITIES

app = FastAPI(title="PNU SmartPark Backend")
DB_NAME = "smartpark.db"

# Defining the Stations (A1 to A6)
STATION_CAPACITIES = {
    "A1": 50, "A2": 60, "A3": 80, 
    "A4": 100, "A5": 40, "A6": 70
}


# 2. LOAD MODELS

try:
    yolo_model = YOLO("best.pt") 
except:
    yolo_model = None

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

try:
    lstm_model = LSTMForecaster()
    lstm_model.load_state_dict(torch.load("pnu_lstm_model.pth", map_location="cpu"))
    lstm_model.eval()
except:
    lstm_model = None


# 3. DATABASE INITIALIZATION

def get_realistic_rate():
    riyadh_tz = pytz.timezone('Asia/Riyadh')
    hr = datetime_.datetime.now(riyadh_tz).hour
    
    if 8 <= hr <= 10: 
        return random.uniform(0.70, 0.90)
    elif 11 <= hr <= 15: 
        return random.uniform(0.30, 0.60)
    else: 
        return random.uniform(0.05, 0.15)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Table for current YOLO detection status
    cursor.execute('''CREATE TABLE IF NOT EXISTS current_status (
        station_id TEXT PRIMARY KEY, last_yolo_count INTEGER DEFAULT 0, 
        total_capacity INTEGER, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Table for user reservations with Start and End times
    cursor.execute('''CREATE TABLE IF NOT EXISTS reservations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, station_id TEXT, 
        spot_index INTEGER, start_time TEXT, end_time TEXT)''')

    for s_id, cap in STATION_CAPACITIES.items():
        cursor.execute("SELECT COUNT(*) FROM current_status WHERE station_id=?", (s_id,))
        if cursor.fetchone()[0] == 0:
            rate = get_realistic_rate()
            initial_yolo = int(cap * rate)
            cursor.execute("INSERT INTO current_status (station_id, last_yolo_count, total_capacity) VALUES (?, ?, ?)", (s_id, initial_yolo, cap))
    conn.commit()
    conn.close()

init_db()


# 4. API ENDPOINTS


@app.get("/api/status")
def get_all_status():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM current_status")
    status_rows = cursor.fetchall()
    
    # Set current time to Riyadh timezone
    import pytz
    from datetime import datetime
    riyadh_tz = pytz.timezone('Asia/Riyadh')
    
    data = []
    for r in status_rows:
        s_id, last_yolo, capacity, _ = r
        
        # Fetch all reservations with start and end times for the station
        cursor.execute("SELECT spot_index, start_time, end_time FROM reservations WHERE station_id=?", (s_id,))
        all_reservations = cursor.fetchall()
        
        active_reserved_indices = []
        for res in all_reservations:
            spot_index, start_time, end_time = res
            
            try:
                # Logic: Determine time format to compare based on frontend input length
                if len(str(start_time)) > 8: 
                    # If frontend sends (Date + Time)
                    current_str = datetime.now(riyadh_tz).strftime("%Y-%m-%d %H:%M:%S")
                else: 
                    # If frontend sends (Time only)
                    current_str = datetime.now(riyadh_tz).strftime("%H:%M")
                
                # Is the current time within the reservation window?
                if str(start_time) <= current_str <= str(end_time):
                    active_reserved_indices.append(spot_index)
            except:
                # Failsafe: If time parsing fails, consider it reserved to prevent double booking
                active_reserved_indices.append(spot_index)
        
        # Calculate totals based only on "active" reservations at this exact moment
        total_occupied = min(last_yolo + len(active_reserved_indices), capacity)
        free_spots = capacity - total_occupied
        
        data.append({
            "station_id": s_id, "yolo_count": last_yolo, "total_capacity": capacity,
            "free_spots": free_spots, "occupied_spots": total_occupied,
            "reserved_indices": active_reserved_indices
        })
    conn.close()
    return {"data": data}



@app.post("/api/reserve")
def reserve_spot(req: ReserveRequest):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Spot-Level Conflict Check 
    cursor.execute("SELECT start_time, end_time FROM reservations WHERE station_id=? AND spot_index=?", (req.station_id, req.spot_index))
    existing_res = cursor.fetchall()
    
    
    req_s = str(req.start_time).split()[-1][:5].zfill(5)
    req_e = str(req.end_time).split()[-1][:5].zfill(5)
    
    for (db_start, db_end) in existing_res:
        db_s = str(db_start).split()[-1][:5].zfill(5)
        db_e = str(db_end).split()[-1][:5].zfill(5)
        
        # Overlap Logic: If (New Start < Existing End) AND (New End > Existing Start) -> Conflict!
        if req_s < db_e and req_e > db_s:
            conn.close()
            raise HTTPException(status_code=400, detail=f"Conflict: Spot {req.spot_index} is already reserved for this time.")
            
    #  Station-Level Capacity Check 
    cursor.execute("SELECT last_yolo_count, total_capacity FROM current_status WHERE station_id=?", (req.station_id,))
    status_row = cursor.fetchone()
    
    if status_row:
        last_yolo, capacity = status_row
        cursor.execute("SELECT COUNT(*) FROM reservations WHERE station_id=?", (req.station_id,))
        res_count = cursor.fetchone()[0]
        
        # If YOLO detected cars + current reservations >= total capacity, reject!
        if (last_yolo + res_count) >= capacity:
            conn.close()
            raise HTTPException(status_code=400, detail="Capacity Error: Station is fully occupied.")
            
    # If all checks pass, save the reservation safely
    cursor.execute("INSERT INTO reservations (student_id, station_id, spot_index, start_time, end_time) VALUES (?, ?, ?, ?, ?)", 
                   (req.student_id, req.station_id, req.spot_index, req.start_time, req.end_time))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/detect/{station_id}")
async def detect_parking(station_id: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        results = yolo_model(img, conf=0.45, iou=0.45)
        # Count cars from YOLO only
        new_yolo = sum(1 for box in results[0].boxes if "not_free" in yolo_model.names[int(box.cls[0])].lower())
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE current_status SET last_yolo_count=?, last_updated=CURRENT_TIMESTAMP WHERE station_id=?", (new_yolo, station_id))
        conn.commit()
        conn.close()
        return {"status": "success", "new_yolo_count": new_yolo}
    except: return {"status": "error"}

@app.get("/api/forecast/{station_id}")
def get_forecast(station_id: str):
    try:
        
        # Get the current exact time in Riyadh
        riyadh_tz = pytz.timezone('Asia/Riyadh')
        current_time = datetime.now(riyadh_tz)
        
        # Dynamically generate the last 16 hours (History)
        history = []
        for i in range(16, 0, -1):
            # Use timedelta to accurately get the past day and hour
            past_time = current_time - timedelta(hours=i)
            h = past_time.hour
            weekday = past_time.weekday() # Monday is 0, Friday is 4, Saturday is 5
            
            # Weekend Logic (Friday and Saturday = Closed)
            if weekday == 4 or weekday == 5:
                val = random.uniform(0.0, 0.05)
            else:
                # Weekday Logic (PNU Pattern)
                if 8 <= h <= 10: 
                    val = random.uniform(0.70, 0.90)  # Morning peak
                elif 11 <= h <= 15: 
                    val = random.uniform(0.30, 0.60)  # Departure cycle
                else: 
                    val = random.uniform(0.05, 0.15)  # Off-peak
                    
            history.append([val])
            
        # 3. Feed the sequence to the LSTM model
        seq = np.array([history])
        tensor = torch.from_numpy(seq).float()
        
        with torch.no_grad(): 
            pred = lstm_model(tensor).item()
            
        # Ensure the prediction is between 0 and 1
        prediction = max(0.0, min(1.0, float(pred)))
        return {"prediction": prediction}
        
    except Exception as e: 
        return {"prediction": 0.0}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
