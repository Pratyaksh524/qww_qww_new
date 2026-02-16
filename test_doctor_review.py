"""
Test script for doctor review API integration
This script demonstrates how to use the send_to_doctor_review method
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.cloud_uploader import get_cloud_uploader

def test_doctor_review_api():
    """Test sending a report to the doctor review API"""
    
    # Get cloud uploader instance
    uploader = get_cloud_uploader()
    
    # Check if doctor review is configured
    if not uploader.doctor_review_enabled:
        print("❌ Doctor review API is not enabled in .env")
        print("   Set DOCTOR_REVIEW_ENABLED=true in .env file")
        return
    
    if not uploader.doctor_review_api_url:
        print("❌ Doctor review API URL is not configured")
        print("   Set DOCTOR_REVIEW_API_URL in .env file")
        return
    
    print(f"✅ Doctor review API is configured")
    print(f"   URL: {uploader.doctor_review_api_url}")
    print(f"   API Key: {'Set' if uploader.doctor_review_api_key else 'Not set'}")
    print()
    
    # Find a sample report to send
    reports_dir = "reports"
    pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF reports found in reports directory")
        return
    
    # Use the most recent report
    pdf_files.sort(reverse=True)
    sample_pdf = os.path.join(reports_dir, pdf_files[0])
    
    print(f"📄 Using sample report: {pdf_files[0]}")
    print()
    
    # Sample patient data
    patient_data = {
        "patient_name": "Test Patient",
        "patient_age": "45",
        "patient_gender": "Male",
        "report_date": "2026-02-13"
    }
    
    # Sample report metadata
    report_metadata = {
        "heart_rate": 75,
        "pr_interval": 160,
        "qrs_duration": 85,
        "qt_interval": 360
    }
    
    print("📤 Sending report to doctor review API...")
    print()
    
    # Send to doctor review
    result = uploader.send_to_doctor_review(
        pdf_path=sample_pdf,
        patient_data=patient_data,
        report_metadata=report_metadata
    )
    
    # Display result
    print("=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message')}")
    
    if result.get('status') == 'success':
        print(f"✅ Report sent successfully!")
        if result.get('api_response'):
            print(f"API Response: {result.get('api_response')}")
    elif result.get('status') == 'queued':
        print(f"📥 Report queued for sending when online")
    else:
        print(f"❌ Failed to send report")
        if result.get('status_code'):
            print(f"Status Code: {result.get('status_code')}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_doctor_review_api()
