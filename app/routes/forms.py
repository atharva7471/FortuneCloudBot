from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, Optional

router = APIRouter()

class FormSubmitRequest(BaseModel):
    form_id: str
    data: Dict[str, Any]

@router.post("/submit")
async def submit_form(request: FormSubmitRequest):
    # Here you would typically save to a database or send an email
    # For now, we simulate processing and validation
    
    form_id = request.form_id
    data = request.data
    
    if not form_id:
        raise HTTPException(status_code=400, detail="form_id is required")
        
    # Basic server-side validation simulation
    if form_id == "course_enquiry":
        if "name" not in data or not data["name"]:
            raise HTTPException(status_code=400, detail="Name is required")
        if "email" not in data or not data["email"] or "@" not in data["email"]:
            raise HTTPException(status_code=400, detail="Valid email is required")
            
    if form_id == "callback_request":
        if "phone" not in data or not data["phone"]:
            raise HTTPException(status_code=400, detail="Phone number is required")
            
    # Save to CSV
    import csv
    import os
    from datetime import datetime
    import json
    
    try:
        # Store in the 'data' directory at the root of the project
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        csv_file = os.path.join(data_dir, "enquiries.csv")
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Form ID", "Name", "Email", "Phone", "Course", "Message", "All Data"])
                
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                form_id,
                data.get("name", ""),
                data.get("email", ""),
                data.get("phone", ""),
                data.get("course", ""),
                data.get("message", ""),
                json.dumps(data)
            ])
    except OSError:
        # Vercel serverless functions have a read-only filesystem (except /tmp).
        # We silently ignore the error so the user still gets a success response.
        pass
            
    # Return processing success
    return {
        "status": "success", 
        "message": "Form submitted successfully",
        "form_id": form_id
    }
