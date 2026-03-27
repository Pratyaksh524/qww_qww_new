"""
Organization Management Module

This module handles organization creation, storage, and role selection functionality
for the ECG Monitor application.
"""

import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QMessageBox, QInputDialog, QLineEdit, QWidget, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from config.settings import resource_path

# File paths for different user types
HEAD_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head_users.json")

def load_head_users():
    """Load head users from separate file"""
    try:
        if os.path.exists(HEAD_USERS_FILE):
            with open(HEAD_USERS_FILE, "r") as f:
                data = json.load(f)
                return data.get("head_users", {})
        else:
            return {}
    except Exception as e:
        print(f"Error loading head users: {e}")
        return {}

def save_head_users(head_users):
    """Save head users to separate file"""
    try:
        data = {"head_users": head_users}
        with open(HEAD_USERS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(head_users)} head users to {HEAD_USERS_FILE}")
    except Exception as e:
        print(f"Error saving head users: {e}")
        raise Exception(f"Failed to save head users: {e}")


class OrganizationManager:
    """Handles organization data management"""
    
    def __init__(self):
        self.organizations_file = resource_path("organizations.json")
    
    def load_organizations(self):
        """Load existing organizations from file"""
        try:
            if os.path.exists(self.organizations_file):
                with open(self.organizations_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading organizations: {e}")
            return {}
    
    def save_organizations(self, organizations):
        """Save organizations to file"""
        try:
            with open(self.organizations_file, "w") as f:
                json.dump(organizations, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving organizations: {e}")
            return False
    
    def add_organization(self, org_name):
        """Add a new organization"""
        if not org_name or not org_name.strip():
            return False, "Organization name cannot be empty"
        
        organizations = self.load_organizations()
        org_name_clean = org_name.strip()
        
        # Check if organization already exists
        if org_name_clean in organizations:
            return False, "Organization already exists"
        
        # Add new organization
        organizations[org_name_clean] = {
            "name": org_name_clean,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }
        
        if self.save_organizations(organizations):
            return True, "Organization added successfully"
        else:
            return False, "Failed to save organization"


class RoleSelectionDialog(QDialog):
    """Dialog for selecting user role after organization creation"""
    
    def __init__(self, parent=None, organization_name="", is_existing_organization=False):
        super().__init__(parent)
        self.organization_name = organization_name
        self.is_existing_organization = is_existing_organization
        self.selected_role = None
        self.selected_organization = organization_name
        self.init_ui()
    
    def init_ui(self):
        """Initialize the role selection dialog UI"""
        self.setWindowTitle("Select Role")
        self.setMinimumSize(400, 350 if self.is_existing_organization else 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background: #ff8800;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Organization: {self.organization_name}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Select your role:")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addStretch(1)
        
        # Initialize login flag
        self.is_login = False
        
        # Role buttons
        doctor_head_btn = QPushButton("Sign up as Head Doctor")
        hcp_head_btn = QPushButton("Sign up as Head HCP")
        
        # Store selected role and organization
        def select_doctor_head():
            self.selected_role = "Doctor Head"
            self.selected_organization = self.organization_name
            # Show sign-up dialog instead of just accepting
            self.show_signup_dialog()
            
        def select_hcp_head():
            self.selected_role = "HCP Head"
            self.selected_organization = self.organization_name
            # Show sign-up dialog instead of just accepting
            self.show_signup_dialog()
        
        doctor_head_btn.clicked.connect(select_doctor_head)
        hcp_head_btn.clicked.connect(select_hcp_head)
        
        layout.addWidget(doctor_head_btn)
        layout.addSpacing(10)
        layout.addWidget(hcp_head_btn)
        
        # Add login options for existing organizations
        if self.is_existing_organization:
            layout.addSpacing(20)
            
            # Add separator line
            separator = QLabel("─ or login as existing user ─")
            separator.setAlignment(Qt.AlignCenter)
            separator.setStyleSheet("color: #ff6600; font-size: 12px; margin: 10px 0;")
            layout.addWidget(separator)
            
            # Create login buttons with smaller size
            login_doctor_btn = QPushButton("Login as Doctor Head")
            login_hcp_btn = QPushButton("Login as HCP Head")
            
            # Style for login buttons (smaller)
            login_button_style = """
                QPushButton {
                    background: rgba(255, 102, 0, 0.8);
                    color: white;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                    min-height: 35px;
                }
                QPushButton:hover {
                    background: rgba(255, 136, 0, 0.9);
                }
            """
            
            login_doctor_btn.setStyleSheet(login_button_style)
            login_hcp_btn.setStyleSheet(login_button_style)
            
            def login_as_doctor_head():
                self.selected_role = "Doctor Head"
                self.selected_organization = self.organization_name
                self.is_login = True  # Flag to indicate login vs signup
                # Show login dialog instead of just accepting
                self.show_login_dialog()
                
            def login_as_hcp_head():
                self.selected_role = "HCP Head"
                self.selected_organization = self.organization_name
                self.is_login = True  # Flag to indicate login vs signup
                # Show login dialog instead of just accepting
                self.show_login_dialog()
            
            login_doctor_btn.clicked.connect(login_as_doctor_head)
            login_hcp_btn.clicked.connect(login_as_hcp_head)
            
            layout.addWidget(login_doctor_btn)
            layout.addSpacing(8)
            layout.addWidget(login_hcp_btn)
        
        layout.addStretch(1)
    
    def get_selection(self):
        """Get the selected role and organization"""
        return self.selected_role, self.selected_organization, getattr(self, 'is_login', False)
    
    def show_signup_dialog(self):
        """Show the sign-up dialog for the selected role"""
        signup_dialog = SignUpDialog(self, self.selected_role, self.selected_organization)
        
        if signup_dialog.exec_() == QDialog.Accepted:
            # Get user data and open dashboard
            user_data = signup_dialog.get_user_data()
            print(f"DEBUG: Signup dialog returned Accepted. User data: {user_data}")
            if user_data:
                self.signup_user_data = user_data
                print(f"DEBUG: signup_user_data set to: {self.signup_user_data}")
                self.accept()  # Close the role selection dialog
                # Open dashboard will be handled by the calling code
            else:
                # Sign-up failed or was cancelled
                print("DEBUG: User data is None after signup")
                self.selected_role = None
                self.selected_organization = None
        else:
            # Sign-up was cancelled
            print("DEBUG: Signup dialog was cancelled or rejected")
            self.selected_role = None
            self.selected_organization = None
    
    def show_login_dialog(self):
        """Show the login dialog for the selected role"""
        login_dialog = LoginDialog(self, self.selected_role, self.selected_organization)
        
        if login_dialog.exec_() == QDialog.Accepted:
            # Get user data and open dashboard
            user_data = login_dialog.get_user_data()
            if user_data:
                self.login_user_data = user_data
                self.accept()  # Close the role selection dialog
                # Open dashboard will be handled by the calling code
            else:
                # Login failed or was cancelled
                self.selected_role = None
                self.selected_organization = None
        else:
            # Login was cancelled
            self.selected_role = None
            self.selected_organization = None


class SignUpDialog(QDialog):
    """Sign-up dialog that matches the provided image design"""
    
    def __init__(self, parent=None, role="", organization=""):
        super().__init__(parent)
        self.role = role
        self.organization = organization
        self.signup_successful = False
        self.user_data = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the sign-up dialog UI"""
        self.setWindowTitle("Sign Up")
        self.setMinimumSize(520, 800)
        self.setModal(True)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                background: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 14px;
                color: #333;
                min-height: 35px;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton#eye_btn {
                background: #6c757d;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-size: 12px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton#eye_btn:hover {
                background: #5a6268;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 50, 40, 40)
        
        # Title
        title = QLabel("Create Account")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: #007bff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Subtitle with role and organization
        subtitle = QLabel(f"{self.role} - {self.organization}")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #6c757d; margin-bottom: 20px;")
        layout.addWidget(subtitle)
        
        # Form fields
        # Full Name
        full_name_layout = QHBoxLayout()
        full_name_label = QLabel("Full Name:")
        full_name_label.setFixedWidth(100)
        full_name_label.setStyleSheet("font-weight: bold; color: #333;")
        self.full_name_edit = QLineEdit()
        self.full_name_edit.setPlaceholderText("Enter your full name")
        self.full_name_edit.setMaximumWidth(300)
        full_name_layout.addWidget(full_name_label)
        full_name_layout.addWidget(self.full_name_edit)
        full_name_layout.addStretch()
        layout.addLayout(full_name_layout)
        layout.addSpacing(8)
        
        # Age
        age_layout = QHBoxLayout()
        age_label = QLabel("Age:")
        age_label.setFixedWidth(100)
        age_label.setStyleSheet("font-weight: bold; color: #333;")
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("Enter your age")
        self.age_edit.setMaximumWidth(150)
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_edit)
        age_layout.addStretch()
        layout.addLayout(age_layout)
        layout.addSpacing(8)
        
        # Gender
        gender_layout = QHBoxLayout()
        gender_label = QLabel("Gender:")
        gender_label.setFixedWidth(100)
        gender_label.setStyleSheet("font-weight: bold; color: #333;")
        self.gender_edit = QLineEdit()
        self.gender_edit.setPlaceholderText("Enter your gender")
        self.gender_edit.setMaximumWidth(200)
        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(self.gender_edit)
        gender_layout.addStretch()
        layout.addLayout(gender_layout)
        layout.addSpacing(8)
        
        # Address
        address_layout = QHBoxLayout()
        address_label = QLabel("Address:")
        address_label.setFixedWidth(100)
        address_label.setStyleSheet("font-weight: bold; color: #333;")
        self.address_edit = QLineEdit()
        self.address_edit.setPlaceholderText("Enter your address")
        self.address_edit.setMaximumWidth(300)
        address_layout.addWidget(address_label)
        address_layout.addWidget(self.address_edit)
        address_layout.addStretch()
        layout.addLayout(address_layout)
        layout.addSpacing(8)
        
        # Phone Number
        phone_layout = QHBoxLayout()
        phone_label = QLabel("Phone Number:")
        phone_label.setFixedWidth(100)
        phone_label.setStyleSheet("font-weight: bold; color: #333;")
        self.phone_edit = QLineEdit()
        self.phone_edit.setPlaceholderText("Enter your phone number")
        self.phone_edit.setMaximumWidth(200)
        phone_layout.addWidget(phone_label)
        phone_layout.addWidget(self.phone_edit)
        phone_layout.addStretch()
        layout.addLayout(phone_layout)
        layout.addSpacing(8)
        
        # Password field with eye toggle
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setFixedWidth(100)
        password_label.setStyleSheet("font-weight: bold; color: #333;")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setMaximumWidth(250)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_edit)
        
        self.password_eye_btn = QPushButton("👁")
        self.password_eye_btn.setObjectName("eye_btn")
        self.password_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.password_edit, self.password_eye_btn))
        password_layout.addWidget(self.password_eye_btn)
        password_layout.addStretch()
        layout.addLayout(password_layout)
        layout.addSpacing(8)
        
        # Confirm password field with eye toggle
        confirm_layout = QHBoxLayout()
        confirm_label = QLabel("Confirm Password:")
        confirm_label.setFixedWidth(100)
        confirm_label.setStyleSheet("font-weight: bold; color: #333;")
        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setPlaceholderText("Confirm password")
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_password_edit.setMaximumWidth(250)
        self.confirm_password_edit.returnPressed.connect(self.handle_signup)
        confirm_layout.addWidget(confirm_label)
        confirm_layout.addWidget(self.confirm_password_edit)
        
        self.confirm_eye_btn = QPushButton("👁")
        self.confirm_eye_btn.setObjectName("eye_btn")
        self.confirm_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.confirm_password_edit, self.confirm_eye_btn))
        confirm_layout.addWidget(self.confirm_eye_btn)
        confirm_layout.addStretch()
        layout.addLayout(confirm_layout)
        layout.addSpacing(20)
        
        # Sign Up button
        self.signup_btn = QPushButton("Sign Up")
        self.signup_btn.clicked.connect(self.handle_signup)
        self.signup_btn.setMaximumWidth(400)
        self.signup_btn.setMinimumHeight(45)
        layout.addWidget(self.signup_btn)
        
        layout.addStretch()
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")
    
    def handle_signup(self):
        """Handle the sign-up process"""
        # Get form data
        full_name = self.full_name_edit.text().strip()
        age = self.age_edit.text().strip()
        gender = self.gender_edit.text().strip()
        address = self.address_edit.text().strip()
        phone = self.phone_edit.text().strip()
        password = self.password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        
        # Validate fields
        if not all([full_name, age, gender, address, phone, password, confirm_password]):
            QMessageBox.warning(self, "Error", "All fields are required.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Error", "Password must be at least 6 characters long.")
            return
        
        # Store user data
        self.user_data = {
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'address': address,
            'phone': phone,
            'password': password,
            'role': self.role,
            'organization': self.organization,
            'signup_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to head_users.json (for Head Doctor and Head HCP roles)
        try:
            print("DEBUG: Attempting to save head user data...")
            head_users = load_head_users()
            username = phone  # Use phone as unique identifier
            
            if username in head_users:
                print(f"DEBUG: Head user {username} already exists")
                QMessageBox.warning(self, "Error", "A user with this phone number already exists.")
                return
            
            head_users[username] = self.user_data
            save_head_users(head_users)
            print("DEBUG: Head user data saved successfully")
            
            self.signup_successful = True
            print("DEBUG: signup_successful set to True")
            QMessageBox.information(self, "Success", "Sign-up successful! Your dashboard will now open.")
            self.accept()
            
        except Exception as e:
            print(f"DEBUG: Exception during head user signup: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save user data: {str(e)}")
    
    def get_user_data(self):
        """Return the signed-up user data"""
        result = self.user_data if self.signup_successful else None
        print(f"DEBUG: get_user_data called. signup_successful={self.signup_successful}, result={result}")
        return result


class LoginDialog(QDialog):
    """Login dialog for existing Head Doctor and Head HCP users"""
    
    def __init__(self, parent=None, role="", organization=""):
        super().__init__(parent)
        self.role = role
        self.organization = organization
        self.login_successful = False
        self.user_data = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the login dialog UI"""
        self.setWindowTitle("Login")
        self.setMinimumSize(450, 400)
        self.setModal(True)
        
        # Set background gradient similar to sign-up dialog
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                background: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 14px;
                color: #333;
                min-height: 35px;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton#eye_btn {
                background: #6c757d;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-size: 12px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton#eye_btn:hover {
                background: #5a6268;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 50, 40, 40)
        
        # Title
        title = QLabel("Login to Account")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: #007bff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Subtitle with role and organization
        subtitle = QLabel(f"{self.role} - {self.organization}")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #6c757d; margin-bottom: 20px;")
        layout.addWidget(subtitle)
        
        # Full Name field
        full_name_layout = QHBoxLayout()
        full_name_label = QLabel("Full Name:")
        full_name_label.setFixedWidth(100)
        full_name_label.setStyleSheet("font-weight: bold; color: #333;")
        self.full_name_edit = QLineEdit()
        self.full_name_edit.setPlaceholderText("Enter your full name")
        self.full_name_edit.setMaximumWidth(300)
        self.full_name_edit.returnPressed.connect(self.handle_login)
        full_name_layout.addWidget(full_name_label)
        full_name_layout.addWidget(self.full_name_edit)
        full_name_layout.addStretch()
        layout.addLayout(full_name_layout)
        layout.addSpacing(8)
        
        # Password field with eye toggle
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setFixedWidth(100)
        password_label.setStyleSheet("font-weight: bold; color: #333;")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter your password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setMaximumWidth(250)
        self.password_edit.returnPressed.connect(self.handle_login)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_edit)
        
        self.password_eye_btn = QPushButton("👁")
        self.password_eye_btn.setObjectName("eye_btn")
        self.password_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.password_edit, self.password_eye_btn))
        password_layout.addWidget(self.password_eye_btn)
        password_layout.addStretch()
        layout.addLayout(password_layout)
        layout.addSpacing(20)
        
        # Login button
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.handle_login)
        self.login_btn.setMaximumWidth(400)
        self.login_btn.setMinimumHeight(45)
        layout.addWidget(self.login_btn)
        
        # Phone login button
        self.phone_login_btn = QPushButton("Login with Phone Number")
        self.phone_login_btn.clicked.connect(self.handle_phone_login)
        self.phone_login_btn.setMaximumWidth(400)
        self.phone_login_btn.setMinimumHeight(40)
        self.phone_login_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5a6268;
            }
        """)
        layout.addWidget(self.phone_login_btn)
        
        layout.addStretch()
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")
    
    def handle_login(self):
        """Handle the login process"""
        full_name = self.full_name_edit.text().strip()
        password = self.password_edit.text()
        
        if not full_name or not password:
            QMessageBox.warning(self, "Error", "Both full name and password are required.")
            return
        
        # Check credentials against head_users.json for Head roles
        try:
            print(f"DEBUG: Looking for full_name='{full_name}', role='{self.role}', org='{self.organization}'")
            head_users = load_head_users()
            print(f"DEBUG: Available head users: {head_users}")
            
            # Find user by full name and role
            found_user = None
            for username, user_record in head_users.items():
                record_full_name = user_record.get('full_name', '').lower()
                record_role = user_record.get('role', '')
                record_org = user_record.get('organization', '')
                input_full_name = full_name.lower()
                
                print(f"DEBUG: Checking head user {username}: full_name='{record_full_name}' vs '{input_full_name}', role='{record_role}' vs '{self.role}', org='{record_org}' vs '{self.organization}'")
                
                if (record_full_name == input_full_name and 
                    record_role == self.role and
                    record_org == self.organization):
                    found_user = user_record
                    print(f"DEBUG: Found matching head user: {found_user}")
                    break
            
            if found_user and found_user.get('password') == password:
                print(f"DEBUG: Password matches for head user")
                self.user_data = found_user
                self.login_successful = True
                QMessageBox.information(self, "Success", "Login successful! Your dashboard will now open.")
                self.accept()
            else:
                if not found_user:
                    print(f"DEBUG: No head user found with matching credentials")
                else:
                    print(f"DEBUG: Password mismatch. Expected: {found_user.get('password')}, Got: {password}")
                QMessageBox.warning(self, "Error", "Invalid credentials. Please check your full name and password.")
                
        except Exception as e:
            print(f"DEBUG: Exception during head user login: {e}")
            QMessageBox.critical(self, "Error", f"Login failed: {str(e)}")
    
    def handle_phone_login(self):
        """Handle phone number login"""
        # Create phone input dialog
        phone_dialog = QDialog(self)
        phone_dialog.setWindowTitle("Login with Phone Number")
        phone_dialog.setMinimumSize(400, 200)
        phone_dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                background: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 14px;
                color: #333;
                min-height: 35px;
            }
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout(phone_dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        label = QLabel("Enter your phone number:")
        layout.addWidget(label)
        
        phone_edit = QLineEdit()
        phone_edit.setPlaceholderText("Enter phone number")
        layout.addWidget(phone_edit)
        
        button_layout = QHBoxLayout()
        login_btn = QPushButton("Login")
        cancel_btn = QPushButton("Cancel")
        
        def do_phone_login():
            phone = phone_edit.text().strip()
            if not phone:
                QMessageBox.warning(phone_dialog, "Error", "Phone number is required.")
                return
            
            # Check credentials against head_users.json
            try:
                head_users = load_head_users()
                print(f"DEBUG: Available head users for phone login: {head_users}")
                
                # Find user by phone and role
                found_user = None
                for username, user_record in head_users.items():
                    if (user_record.get('phone', '') == phone and 
                        user_record.get('role', '') == self.role and
                        user_record.get('organization', '') == self.organization):
                        found_user = user_record
                        break
                
                if found_user:
                    self.user_data = found_user
                    self.login_successful = True
                    phone_dialog.accept()
                    self.accept()
                else:
                    QMessageBox.warning(phone_dialog, "Error", "No account found with this phone number.")
                    
            except Exception as e:
                QMessageBox.critical(phone_dialog, "Error", f"Login failed: {str(e)}")
        
        login_btn.clicked.connect(do_phone_login)
        cancel_btn.clicked.connect(phone_dialog.reject)
        
        button_layout.addWidget(login_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        if phone_dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Success", "Login successful! Your dashboard will now open.")
    
    def get_user_data(self):
        """Return the logged-in user data"""
        return self.user_data if self.login_successful else None


class DashboardWindow(QDialog):
    """Full-screen dashboard that opens after successful sign-up"""
    
    def __init__(self, parent=None, user_data=None):
        super().__init__(parent)
        self.user_data = user_data or {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the dashboard UI"""
        self.setWindowTitle("Dashboard")
        self.showMaximized()  # Full screen
        self.setModal(False)  # Non-modal so user can interact with other windows if needed
        
        # Enable window flags for better window management
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QFrame {
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.2);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.user_data.get('full_name', 'User')}!")
        welcome_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ff6600;")
        header_layout.addWidget(welcome_label)
        
        header_layout.addStretch()
        
        # Role and organization info
        role_org_label = QLabel(f"{self.user_data.get('role', 'N/A')} - {self.user_data.get('organization', 'N/A')}")
        role_org_label.setStyleSheet("font-size: 16px; color: #ccc;")
        header_layout.addWidget(role_org_label)
        
        # Logout button
        logout_btn = QPushButton("Logout")
        logout_btn.clicked.connect(self.close)
        header_layout.addWidget(logout_btn)
        
        layout.addLayout(header_layout)
        
        # Main content area
        main_content = QHBoxLayout()
        
        # Left sidebar - User info card
        left_frame = QFrame()
        left_frame.setFixedWidth(300)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # User info title
        user_info_title = QLabel("User Information")
        user_info_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        left_layout.addWidget(user_info_title)
        
        # User details
        user_details = [
            ("Full Name:", self.user_data.get('full_name', 'N/A')),
            ("Age:", self.user_data.get('age', 'N/A')),
            ("Gender:", self.user_data.get('gender', 'N/A')),
            ("Phone:", self.user_data.get('phone', 'N/A')),
            ("Address:", self.user_data.get('address', 'N/A')),
            ("Role:", self.user_data.get('role', 'N/A')),
            ("Organization:", self.user_data.get('organization', 'N/A')),
            ("Member Since:", self.user_data.get('signup_date', 'N/A'))
        ]
        
        for label, value in user_details:
            detail_layout = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: bold; color: #ff6600;")
            value_widget = QLabel(value)
            value_widget.setWordWrap(True)
            detail_layout.addWidget(label_widget)
            detail_layout.addWidget(value_widget)
            detail_layout.addStretch()
        
        # Add Users button for Doctor Head and HCP Head
        if self.user_data.get('role') in ['Doctor Head', 'HCP Head']:
            add_users_btn = QPushButton("Add Users")
            add_users_btn.setStyleSheet("""
                QPushButton {
                    background: #28a745;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #218838;
                }
            """)
            add_users_btn.clicked.connect(self.show_add_users_dialog)
            left_layout.addWidget(add_users_btn)
            left_layout.addSpacing(10)
            
            # User count display for Doctor Head and HCP Head
            if self.user_data.get('role') in ['Doctor Head', 'HCP Head']:
                user_count = self.get_user_count()
                count_label = QLabel(f"Users: {user_count}")
                count_label.setStyleSheet("""
                    QLabel {
                        background: rgba(40, 167, 69, 0.2);
                        color: #28a745;
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                count_label.setAlignment(Qt.AlignCenter)
                left_layout.addWidget(count_label)
        
        left_layout.addStretch()
        
        main_content.addWidget(left_frame)
        
        # Center area - Main dashboard content
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(30, 30, 30, 30)
        
        # Dashboard title
        dashboard_title = QLabel("Dashboard Overview")
        dashboard_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        center_layout.addWidget(dashboard_title)
        
        # Quick stats
        stats_layout = QHBoxLayout()
        
        # Sample stats cards
        stats_data = [
            ("Total Patients", "0", "#007bff"),
            ("Appointments Today", "0", "#28a745"),
            ("Pending Reports", "0", "#ffc107"),
            ("Messages", "0", "#dc3545")
        ]
        
        for title, value, color in stats_data:
            stat_card = QFrame()
            stat_card.setStyleSheet(f"""
                QFrame {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 8px;
                    border: none;
                }}
                QLabel {{
                    color: white;
                }}
            """)
            stat_card.setMinimumWidth(200)
            stat_layout_inner = QVBoxLayout(stat_card)
            stat_layout_inner.setContentsMargins(15, 15, 15, 15)
            
            stat_value = QLabel(value)
            stat_value.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color};")
            stat_value.setAlignment(Qt.AlignCenter)
            stat_layout_inner.addWidget(stat_value)
            
            stat_title = QLabel(title)
            stat_title.setStyleSheet("font-size: 14px; margin-top: 5px;")
            stat_title.setAlignment(Qt.AlignCenter)
            stat_layout_inner.addWidget(stat_title)
            
            stats_layout.addWidget(stat_card)
        
        center_layout.addLayout(stats_layout)
        center_layout.addSpacing(30)
        
        # Recent activity section
        activity_title = QLabel("Recent Activity")
        activity_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        center_layout.addWidget(activity_title)
        
        # Activity placeholder
        activity_placeholder = QLabel("No recent activity to display.")
        activity_placeholder.setStyleSheet("color: #ccc; font-style: italic; padding: 20px;")
        activity_placeholder.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(activity_placeholder)
        
        center_layout.addStretch()
        main_content.addWidget(center_frame, 1)
        
        layout.addLayout(main_content)
        
        # Footer
        footer_label = QLabel("© 2024 CardioX by Deckmount. All rights reserved.")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: #666; font-size: 12px; margin-top: 20px;")
        layout.addWidget(footer_label)
    
    def show_add_users_dialog(self):
        """Show dialog for adding users based on role hierarchy"""
        dialog = AddUsersDialog(self, self.user_data)
        dialog.exec_()
    
    def get_user_count(self):
        """Get count of users in the same organization"""
        try:
            from main import load_users
            users = load_users()
            current_org = self.user_data.get('organization', '')
            count = 0
            
            for username, user_data in users.items():
                if user_data.get('organization') == current_org:
                    count += 1
            
            return count
        except Exception:
            return 0


class OrganizationRequestHandler:
    """Handles organization request workflow"""
    
    def __init__(self, parent_dialog):
        self.parent_dialog = parent_dialog
        self.org_manager = OrganizationManager()
    
    def handle_organization_request(self):
        """Handle request for new organization name"""
        org_name, ok = QInputDialog.getText(
            self.parent_dialog, 
            "Request New Organization", 
            "Enter organization name:",
            QLineEdit.Normal
        )
        
        if ok and org_name.strip():
            # Add the organization
            success, message = self.org_manager.add_organization(org_name.strip())
            
            if success:
                QMessageBox.information(self.parent_dialog, "Success", "New organization added successfully!")
                
                # Show role selection dialog
                self.show_role_selection_dialog(org_name.strip())
            else:
                QMessageBox.warning(self.parent_dialog, "Error", message)
    
    def handle_existing_organization_request(self):
        """Handle request for existing organization name"""
        # Get list of existing organizations
        organizations = self.org_manager.load_organizations()
        
        if not organizations:
            QMessageBox.warning(self.parent_dialog, "No Organizations", 
                              "No existing organizations found. Please request a new organization first.")
            return
        
        # Create a dialog to select existing organization
        dialog = QDialog(self.parent_dialog)
        dialog.setWindowTitle("Select Existing Organization")
        dialog.setMinimumSize(400, 300)
        dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QListWidget {
                background: rgba(255,255,255,0.1);
                border: 1px solid #ff6600;
                border-radius: 8px;
                padding: 5px;
                color: white;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #ff6600;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("Select Existing Organization")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Organization list with delete functionality
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QHBoxLayout
        org_list = QListWidget()
        
        for org_name in organizations.keys():
            # Create custom widget for each organization item
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 5, 5, 5)
            
            # Organization name label
            org_label = QLabel(org_name)
            org_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
            item_layout.addWidget(org_label)
            
            # Add stretch to push delete button to the right
            item_layout.addStretch()
            
            # Delete button
            delete_btn = QPushButton("X")
            delete_btn.setFixedSize(30, 30)
            delete_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 0, 0, 0.8);
                    border: none;
                    border-radius: 15px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(255, 0, 0, 1.0);
                }
            """)
            
            # Connect delete button to delete function
            def delete_organization(org=org_name, dlg=dialog):
                reply = QMessageBox.question(
                    dlg, 
                    "Confirm Delete", 
                    f"Are you sure you want to delete the organization '{org}'?\n\nThis action cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Remove organization from the organizations dict
                    organizations.pop(org, None)
                    # Save updated organizations
                    if self.org_manager.save_organizations(organizations):
                        QMessageBox.information(dlg, "Success", f"Organization '{org}' deleted successfully.")
                        # Clear and rebuild the organization list
                        org_list.clear()
                        for org_name in organizations.keys():
                            # Create custom widget for each organization item
                            item_widget = QWidget()
                            item_layout = QHBoxLayout(item_widget)
                            item_layout.setContentsMargins(5, 5, 5, 5)
                            
                            # Organization name label
                            org_label = QLabel(org_name)
                            org_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
                            item_layout.addWidget(org_label)
                            
                            # Add stretch to push delete button to the right
                            item_layout.addStretch()
                            
                            # Delete button
                            delete_btn = QPushButton("X")
                            delete_btn.setFixedSize(30, 30)
                            delete_btn.setStyleSheet("""
                                QPushButton {
                                    background: rgba(255, 0, 0, 0.8);
                                    border: none;
                                    border-radius: 15px;
                                    color: white;
                                    font-size: 16px;
                                    font-weight: bold;
                                }
                                QPushButton:hover {
                                    background: rgba(255, 0, 0, 1.0);
                                }
                            """)
                            
                            # Connect delete button to delete function
                            def delete_organization_new(org_new=org_name, dlg_new=dlg):
                                reply_new = QMessageBox.question(
                                    dlg_new, 
                                    "Confirm Delete", 
                                    f"Are you sure you want to delete the organization '{org_new}'?\n\nThis action cannot be undone.",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No
                                )
                                
                                if reply_new == QMessageBox.Yes:
                                    # Remove organization from the organizations dict
                                    organizations.pop(org_new, None)
                                    # Save updated organizations
                                    if self.org_manager.save_organizations(organizations):
                                        QMessageBox.information(dlg_new, "Success", f"Organization '{org_new}' deleted successfully.")
                                        # Remove the item from the list
                                        for i in range(org_list.count()):
                                            item = org_list.item(i)
                                            widget = org_list.itemWidget(item)
                                            if widget:
                                                # Find the organization label in the widget
                                                for child in widget.children():
                                                    if isinstance(child, QLabel) and child.text() == org_new:
                                                        org_list.takeItem(i)
                                                        return
                                    else:
                                        QMessageBox.warning(dlg_new, "Error", "Failed to delete organization.")
                            
                            delete_btn.clicked.connect(delete_organization_new)
                            item_layout.addWidget(delete_btn)
                            
                            # Create list item and set the custom widget
                            list_item = QListWidgetItem(org_list)
                            list_item.setSizeHint(item_widget.sizeHint())
                            org_list.setItemWidget(list_item, item_widget)
                    else:
                        QMessageBox.warning(dlg, "Error", "Failed to delete organization.")
            
            delete_btn.clicked.connect(delete_organization)
            item_layout.addWidget(delete_btn)
            
            # Create list item and set the custom widget
            list_item = QListWidgetItem(org_list)
            list_item.setSizeHint(item_widget.sizeHint())
            org_list.setItemWidget(list_item, item_widget)
        
        layout.addWidget(org_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        
        def select_organization():
            current_item = org_list.currentItem()
            if current_item:
                # Get the widget from the list item to extract organization name
                widget = org_list.itemWidget(current_item)
                if widget:
                    # Find the organization label in the widget
                    for child in widget.children():
                        if isinstance(child, QLabel) and "X" not in child.text():
                            selected_org = child.text()
                            dialog.accept()
                            # Show role selection dialog
                            self.show_role_selection_dialog(selected_org, is_existing_organization=True)
                            return
                QMessageBox.warning(dialog, "Warning", "Please select an organization.")
            else:
                QMessageBox.warning(dialog, "Warning", "Please select an organization.")
        
        select_btn.clicked.connect(select_organization)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(select_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Execute dialog
        dialog.exec_()
    
    def show_role_selection_dialog(self, organization_name, is_existing_organization=False):
        """Show role selection dialog with Doctor Head and HCP Head options"""
        dialog = RoleSelectionDialog(self.parent_dialog, organization_name, is_existing_organization)
        
        if dialog.exec_() == QDialog.Accepted:
            role, org, is_login = dialog.get_selection()
            
            # Store the selection in the parent dialog
            self.parent_dialog.selected_role = role
            self.parent_dialog.selected_organization = org
            
            if is_login:
                # Handle login for existing user - check if login was completed
                if hasattr(dialog, 'login_user_data') and dialog.login_user_data:
                    # Login was successful, open dashboard
                    QMessageBox.information(self.parent_dialog, "Login Complete", 
                        f"Login successful!\nRole: {role}\nOrganization: {org}\n\nOpening your dashboard...")
                    
                    # Open the dashboard
                    self.open_dashboard(dialog.login_user_data)
                else:
                    # Login was cancelled or failed
                    QMessageBox.information(self.parent_dialog, "Login Cancelled", 
                        "Login was cancelled or incomplete.")
            else:
                # Handle signup for new user - check if signup was completed
                if hasattr(dialog, 'signup_user_data') and dialog.signup_user_data:
                    # Sign-up was successful, open dashboard
                    QMessageBox.information(self.parent_dialog, "Registration Complete", 
                        f"Registration successful!\nRole: {role}\nOrganization: {org}\n\nOpening your dashboard...")
                    
                    # Open the dashboard
                    self.open_dashboard(dialog.signup_user_data)
                else:
                    # Sign-up was cancelled or failed
                    QMessageBox.information(self.parent_dialog, "Registration Cancelled", 
                        "Registration was cancelled or incomplete.")
            
            return role, org, is_login
        
        return None, None, None
    
    def open_dashboard(self, user_data):
        """Open the dashboard window with user data"""
        try:
            dashboard = DashboardWindow(self.parent_dialog, user_data)
            dashboard.show()
            # Bring dashboard to front
            dashboard.raise_()
            dashboard.activateWindow()
        except Exception as e:
            QMessageBox.critical(self.parent_dialog, "Error", f"Failed to open dashboard: {str(e)}")


def create_organization_request_button(parent_dialog):
    """Create and return the organization request button with connected handler"""
    org_request_btn = QPushButton("Request for New Organization Name")
    org_request_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #ff6600;
            border: none;
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 14px;
            font-weight: bold;
            text-decoration: underline;
        }
        QPushButton:hover {
            background: rgba(255, 102, 0, 0.1);
        }
    """)
    
    # Create handler and connect button
    handler = OrganizationRequestHandler(parent_dialog)
    org_request_btn.clicked.connect(handler.handle_organization_request)
    
    return org_request_btn, handler


def create_existing_organization_button(parent_dialog):
    """Create and return the existing organization button with connected handler"""
    existing_org_btn = QPushButton("Existing Organization")
    existing_org_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #ff6600;
            border: none;
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 14px;
            font-weight: bold;
            text-decoration: underline;
        }
        QPushButton:hover {
            background: rgba(255, 102, 0, 0.1);
        }
    """)
    
    # Create handler and connect button
    handler = OrganizationRequestHandler(parent_dialog)
    existing_org_btn.clicked.connect(handler.handle_existing_organization_request)
    
    return existing_org_btn, handler


def create_organization_buttons_layout(parent_dialog):
    """Create a layout with both organization buttons and return the layout and handlers"""
    from PyQt5.QtWidgets import QHBoxLayout
    
    # Create buttons layout
    buttons_layout = QHBoxLayout()
    
    # Create both buttons
    new_org_btn, new_handler = create_organization_request_button(parent_dialog)
    existing_org_btn, existing_handler = create_existing_organization_button(parent_dialog)
    
    # Add buttons to layout
    buttons_layout.addWidget(new_org_btn)
    buttons_layout.addWidget(existing_org_btn)
    
    return buttons_layout, new_handler, existing_handler


class AddUsersDialog(QDialog):
    """Dialog for adding users based on role hierarchy"""
    
    def __init__(self, parent, current_user_data):
        super().__init__(parent)
        self.current_user_data = current_user_data
        self.current_role = current_user_data.get('role', '')
        self.init_ui()
    
    def init_ui(self):
        """Initialize the add users dialog UI"""
        self.setWindowTitle("Add Users")
        self.setMinimumSize(700, 600)  # Made dialog bigger
        self.setModal(True)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton#add_btn {
                background: #28a745;
            }
            QPushButton#add_btn:hover {
                background: #218838;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel(f"Add Users - {self.current_role}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: #007bff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Role selection based on current user
        if self.current_role == "Doctor Head":
            # Doctor Head can add Clinical Users (Sr. Clinical Doctor, Jr. Clinical Doctor)
            subtitle = QLabel("Select user type to add:")
            subtitle.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
            layout.addWidget(subtitle)
            
            # Clinical User buttons
            clinical_layout = QHBoxLayout()
            
            sr_clinical_btn = QPushButton("Add Sr. Clinical Doctor")
            sr_clinical_btn.setObjectName("add_btn")
            sr_clinical_btn.clicked.connect(lambda: self.show_user_form("Sr. Clinical Doctor"))
            clinical_layout.addWidget(sr_clinical_btn)
            
            jr_clinical_btn = QPushButton("Add Jr. Clinical Doctor")
            jr_clinical_btn.setObjectName("add_btn")
            jr_clinical_btn.clicked.connect(lambda: self.show_user_form("Jr. Clinical Doctor"))
            clinical_layout.addWidget(jr_clinical_btn)
            
            layout.addLayout(clinical_layout)
            
        elif self.current_role == "HCP Head":
            # HCP Head can add Admin Users (Sr. Admin, Jr. Admin) and Sub Dealer roles
            subtitle = QLabel("Select user type to add:")
            subtitle.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
            layout.addWidget(subtitle)
            
            # Admin User buttons
            admin_layout = QHBoxLayout()
            
            sr_admin_btn = QPushButton("Add Sr. Admin")
            sr_admin_btn.setObjectName("add_btn")
            sr_admin_btn.clicked.connect(lambda: self.show_user_form("Sr. Admin"))
            admin_layout.addWidget(sr_admin_btn)
            
            jr_admin_btn = QPushButton("Add Jr. Admin")
            jr_admin_btn.setObjectName("add_btn")
            jr_admin_btn.clicked.connect(lambda: self.show_user_form("Jr. Admin"))
            admin_layout.addWidget(jr_admin_btn)
            
            layout.addLayout(admin_layout)
            
            layout.addSpacing(20)
            
            # Sub Dealer buttons
            subtitle2 = QLabel("Or add Sub Dealer roles:")
            subtitle2.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
            layout.addWidget(subtitle2)
            
            sub_dealer_layout = QHBoxLayout()
            
            employee_btn = QPushButton("Add Employee")
            employee_btn.setObjectName("add_btn")
            employee_btn.clicked.connect(lambda: self.show_user_form("Employee"))
            sub_dealer_layout.addWidget(employee_btn)
            
            receptionist_btn = QPushButton("Add Receptionist")
            receptionist_btn.setObjectName("add_btn")
            receptionist_btn.clicked.connect(lambda: self.show_user_form("Receptionist"))
            sub_dealer_layout.addWidget(receptionist_btn)
            
            layout.addLayout(sub_dealer_layout)
        
        layout.addStretch()
    
    def show_user_form(self, user_role):
        """Show user creation form for the selected role"""
        form_dialog = UserCreationDialog(self, user_role, self.current_user_data)
        form_dialog.exec_()
    
    def get_user_data(self):
        """Return the created user data"""
        return self.user_data if hasattr(self, 'user_data') else None


class UserCreationDialog(QDialog):
    """Dialog for creating new users"""
    
    def __init__(self, parent, user_role, current_user_data):
        super().__init__(parent)
        self.user_role = user_role
        self.current_user_data = current_user_data
        self.user_data = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user creation dialog UI"""
        self.setWindowTitle(f"Create {self.user_role}")
        self.setMinimumSize(700, 750)  # Made dialog bigger
        self.setModal(True)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                background: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 14px;
                color: #333;
                min-height: 35px;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton#create_btn {
                background: #28a745;
            }
            QPushButton#create_btn:hover {
                background: #218838;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel(f"Create {self.user_role}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: #007bff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Form fields
        fields = [
            ("Full Name:", "full_name", "Enter full name"),
            ("Age:", "age", "Enter age"),
            ("Gender:", "gender", "Enter gender"),
            ("Address:", "address", "Enter address"),
            ("Phone Number:", "phone", "Enter phone number"),
            ("Password:", "password", "Enter password"),
            ("Confirm Password:", "confirm_password", "Confirm password")
        ]
        
        self.field_widgets = {}
        
        for label_text, field_name, placeholder in fields:
            field_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            label.setStyleSheet("font-weight: bold; color: #333;")
            
            if field_name in ['password', 'confirm_password']:
                field = QLineEdit()
                field.setPlaceholderText(placeholder)
                field.setEchoMode(QLineEdit.Password)
                field.setMaximumWidth(250)  # Made input box smaller
                field.returnPressed.connect(self.handle_create_user)
            else:
                field = QLineEdit()
                field.setPlaceholderText(placeholder)
                field.setMaximumWidth(250)  # Made input box smaller
                if field_name == 'confirm_password':
                    field.returnPressed.connect(self.handle_create_user)
            
            self.field_widgets[field_name] = field
            field_layout.addWidget(label)
            field_layout.addWidget(field)
            field_layout.addStretch()
            layout.addLayout(field_layout)
            layout.addSpacing(8)
        
        # Create button
        create_btn = QPushButton("Create User")
        create_btn.setObjectName("create_btn")
        create_btn.clicked.connect(self.handle_create_user)
        layout.addWidget(create_btn)
        
        # Existing User link
        existing_user_link = QPushButton("Existing User")
        existing_user_link.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #007bff;
                border: none;
                text-decoration: underline;
                font-size: 12px;
                padding: 5px;
            }
            QPushButton:hover {
                color: #0056b3;
                background: rgba(0, 123, 255, 0.1);
            }
        """)
        existing_user_link.clicked.connect(self.handle_existing_user)
        layout.addWidget(existing_user_link)
        
        layout.addStretch()
    
    def handle_create_user(self):
        """Handle user creation"""
        # Get form data
        full_name = self.field_widgets['full_name'].text().strip()
        age = self.field_widgets['age'].text().strip()
        gender = self.field_widgets['gender'].text().strip()
        address = self.field_widgets['address'].text().strip()
        phone = self.field_widgets['phone'].text().strip()
        password = self.field_widgets['password'].text()
        confirm_password = self.field_widgets['confirm_password'].text()
        
        # Validation
        if not full_name or not password:
            QMessageBox.warning(self, "Error", "Full name and password are required.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        
        try:
            # Import user management functions
            from main import load_users, save_users
            
            users = load_users()
            
            # Create new user record
            new_user = {
                'full_name': full_name,
                'age': age,
                'gender': gender,
                'address': address,
                'phone': phone,
                'password': password,
                'role': self.user_role,
                'organization': self.current_user_data.get('organization', ''),
                'signup_date': f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'created_by': self.current_user_data.get('full_name', '')
            }
            
            # Generate username from phone or use full name
            username = phone if phone else full_name.replace(' ', '_').lower()
            users[username] = new_user
            
            # Save users
            save_users(users)
            
            self.user_data = new_user
            QMessageBox.information(self, "Success", f"{self.user_role} created successfully!")
            
            # Open user management dashboard
            self.open_user_management_dashboard()
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create user: {str(e)}")
    
    def handle_existing_user(self):
        """Handle existing user link - close all dialogs and go to login"""
        # Close all dialogs and go to login
        self.parent().parent().close()  # Close AddUsersDialog
        self.parent().close()  # Close DashboardWindow
        # The main application will return to login screen
    
    def open_user_management_dashboard(self):
        """Open user management dashboard with user count"""
        user_mgmt_dialog = UserManagementDashboard(self, self.current_user_data)
        user_mgmt_dialog.exec_()
    
    def get_user_data(self):
        """Return the created user data"""
        return self.user_data if hasattr(self, 'user_data') else None


class UserManagementDashboard(QDialog):
    """Dashboard for managing users with count display"""
    
    def __init__(self, parent, current_user_data):
        super().__init__(parent)
        self.current_user_data = current_user_data
        self.init_ui()
    
    def init_ui(self):
        """Initialize user management dashboard UI"""
        self.setWindowTitle("User Management")
        self.setMinimumSize(900, 600)
        self.setModal(True)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QFrame {
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.2);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("User Management Dashboard")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(header_label)
        
        # User count section
        count_frame = QFrame()
        count_layout = QVBoxLayout(count_frame)
        count_layout.setContentsMargins(20, 20, 20, 20)
        
        # Calculate user count
        user_count = self.get_user_count()
        
        count_label = QLabel(f"Total Users: {user_count}")
        count_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #28a745; margin-bottom: 10px;")
        count_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(count_label)
        
        subtitle_label = QLabel("Users created under your organization")
        subtitle_label.setStyleSheet("font-size: 14px; color: #ccc;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(subtitle_label)
        
        layout.addWidget(count_frame)
        
        # Action buttons
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        
        back_btn = QPushButton("Back to Dashboard")
        back_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(back_btn)
        
        add_more_btn = QPushButton("Add More Users")
        add_more_btn.clicked.connect(self.add_more_users)
        buttons_layout.addWidget(add_more_btn)
        
        layout.addWidget(buttons_frame)
        layout.addStretch()
    
    def get_user_count(self):
        """Get count of users in the same organization"""
        try:
            from main import load_users
            users = load_users()
            current_org = self.current_user_data.get('organization', '')
            count = 0
            
            for username, user_data in users.items():
                if user_data.get('organization') == current_org:
                    count += 1
            
            return count
        except Exception:
            return 0
    
    def add_more_users(self):
        """Open add users dialog again"""
        self.accept()
        add_dialog = AddUsersDialog(self.parent(), self.current_user_data)
        add_dialog.exec_()
