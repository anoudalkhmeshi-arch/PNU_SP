from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import sqlite3
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import random


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
    hr = datetime.now().hour
    if 8 <= hr <= 10: return random.uniform(0.70, 0.90)
    elif 11 <= hr <= 15: return random.uniform(0.30, 0.60)
    else: return random.uniform(0.05, 0.15)

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
    
    data = []
    for r in status_rows:
        s_id, last_yolo, capacity, _ = r
        # Get active reservations
        cursor.execute("SELECT spot_index FROM reservations WHERE station_id=?", (s_id,))
        reserved_indices = [row[0] for row in cursor.fetchall()]
        
        # Cumulative Logic: Occupied = YOLO count + Reservation count
        total_occupied = min(last_yolo + len(reserved_indices), capacity)
        free_spots = capacity - total_occupied
        
        data.append({
            "station_id": s_id, "yolo_count": last_yolo, "total_capacity": capacity,
            "free_spots": free_spots, "occupied_spots": total_occupied,
            "reserved_indices": reserved_indices
        })
    conn.close()
    return {"data": data}

class ReserveRequest(BaseModel):
    student_id: str
    station_id: str
    spot_index: int
    start_time: str
    end_time: str

@app.post("/api/reserve")
def reserve_spot(req: ReserveRequest):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
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
        seq = np.full((1, 16, 1), 0.5)
        tensor = torch.from_numpy(seq).float()
        with torch.no_grad(): pred = lstm_model(tensor).item()
        return {"prediction": float(pred)}
    except: return {"prediction": 0.0}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)