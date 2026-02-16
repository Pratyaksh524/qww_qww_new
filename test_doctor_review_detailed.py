"""
Detailed test for doctor review API
Shows full request and response details
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.cloud_uploader import get_cloud_uploader
import json

def detailed_test():
    uploader = get_cloud_uploader()
    
    print("="*60)
    print("DOCTOR REVIEW API - DETAILED TEST")
    print("="*60)
    print()
    
    # Configuration
    print("📋 Configuration:")
    print(f"   Enabled: {uploader.doctor_review_enabled}")
    print(f"   URL: {uploader.doctor_review_api_url}")
    print(f"   API Key: {'Set' if uploader.doctor_review_api_key else 'Not set'}")
    print()
    
    # Find a sample report
    reports_dir = "reports"
    pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF reports found")
        return
    
    pdf_files.sort(reverse=True)
    sample_pdf = os.path.join(reports_dir, pdf_files[0])
    
    print(f"📄 Test Report: {pdf_files[0]}")
    print()
    
    # Sample data
    patient_data = {
        "patient_name": "Test Patient",
        "patient_age": "45",
        "patient_gender": "Male",
        "report_date": "2026-02-13"
    }
    
    report_metadata = {
        "heart_rate": 75,
        "pr_interval": 160,
        "qrs_duration": 85
    }
    
    print("📤 Sending request...")
    print(f"   Patient Data: {json.dumps(patient_data, indent=2)}")
    print(f"   Metadata: {json.dumps(report_metadata, indent=2)}")
    print()
    
    # Send request
    result = uploader.send_to_doctor_review(
        pdf_path=sample_pdf,
        patient_data=patient_data,
        report_metadata=report_metadata
    )
    
    # Display result
    print("="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)

if __name__ == "__main__":
    detailed_test()
