import streamlit as st
import requests
import time
import json
import uuid
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Page config MUST be first
st.set_page_config(
    page_title="GET Holdings - Support Desk",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Firebase Configuration
try:
    from config import FIREBASE_WEB_API_KEY, FIREBASE_DB_URL, GEMINI_API_KEY, ADMIN_REGISTRATION_CODE
except ImportError:
    st.error("❌ Config file not found. Please create config.py with your Firebase and Gemini API keys.")
    st.stop()

# PDF Generation Setup - Using WeasyPrint as alternative
PDF_AVAILABLE = False
PDF_ENGINE = None

try:
    import weasyprint
    from jinja2 import Template
    PDF_AVAILABLE = True
    PDF_ENGINE = "weasyprint"
    print("✅ Using WeasyPrint for PDF generation")
except ImportError:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_pdf import PdfPages
        PDF_AVAILABLE = True
        PDF_ENGINE = "matplotlib"
        print("✅ Using Matplotlib for PDF generation")
    except ImportError:
        try:
            import pdfkit
            PDF_AVAILABLE = True
            PDF_ENGINE = "pdfkit"
            print("✅ Using pdfkit for PDF generation")
        except ImportError:
            print("❌ No PDF library available")

# Improved Gemini AI Setup with better error handling
AI_AVAILABLE = False
working_model_name = None
model = None

def test_gemini_connection():
    """Test Gemini API connection with detailed debugging"""
    try:
        import google.generativeai as genai
        
        # Check if API key exists
        if not GEMINI_API_KEY or GEMINI_API_KEY == "":
            print("❌ GEMINI_API_KEY is not set in config.py")
            return False, None, None
        
        # Configure API key
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"✅ API key configured (first 10 chars): {GEMINI_API_KEY[:10]}...")
        
        # Model candidates in order of preference
        model_candidates = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest',
            'gemini-1.0-pro'
        ]
        
        for candidate in model_candidates:
            try:
                print(f"🔄 Testing model: {candidate}")
                
                # Create model instance with safety settings
                model = genai.GenerativeModel(
                    candidate,
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    }
                )
                
                # Test with a simple prompt
                response = model.generate_content(
                    "Hello, please respond with exactly: 'AI system working correctly'",
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=50,
                        temperature=0.1,
                    )
                )
                
                if response and response.text and "AI system working correctly" in response.text:
                    print(f"✅ Successfully connected to {candidate}")
                    print(f"✅ Test response: {response.text}")
                    return True, candidate, model
                else:
                    print(f"❌ Model {candidate} returned unexpected response: {response.text if response else 'None'}")
                    continue
                    
            except Exception as e:
                print(f"❌ Model {candidate} failed: {str(e)}")
                continue
        
        print("❌ All Gemini models failed to initialize")
        return False, None, None
        
    except ImportError:
        print("❌ google-generativeai package not installed. Run: pip install google-generativeai")
        return False, None, None
    except Exception as e:
        print(f"❌ Gemini AI setup failed: {str(e)}")
        return False, None, None

# Initialize Gemini AI
try:
    AI_AVAILABLE, working_model_name, model = test_gemini_connection()
    if AI_AVAILABLE:
        print(f"🤖 Gemini AI Ready: {working_model_name}")
    else:
        print("🤖 Gemini AI Not Available")
except Exception as e:
    print(f"🤖 AI initialization error: {e}")
    AI_AVAILABLE = False

# ========== Enhanced Knowledge Base Functions with AI ==========
def create_knowledge_article(article_data):
    """Create new knowledge base article"""
    article_id = str(uuid.uuid4())
    article_data['article_id'] = article_id
    article_data['created_at'] = int(time.time())
    article_data['updated_at'] = int(time.time())
    article_data['ai_generated'] = article_data.get('ai_generated', False)
    
    url = f"{FIREBASE_DB_URL}/knowledge_base/{article_id}.json"
    response = requests.put(url, json=article_data)
    
    if response.status_code == 200:
        notify_all_admins(f"📚 New knowledge article created: {article_data['title']}")
        log_activity("knowledge_created", "system", f"Created knowledge article: {article_data['title']}")
    
    return response.status_code == 200

def get_knowledge_base():
    """Get all knowledge base articles"""
    url = f"{FIREBASE_DB_URL}/knowledge_base.json"
    response = requests.get(url)
    
    if response.status_code == 200 and response.json():
        return response.json()
    return {}

def ai_generate_knowledge_article(topic, context=""):
    """Generate knowledge base article using AI"""
    if not AI_AVAILABLE:
        return None
    
    try:
        prompt = f"""
        You are a technical documentation expert for GET Holdings (Identity Management & Dubai Operations).
        
        Generate a comprehensive knowledge base article for: "{topic}"
        Additional context: {context}
        
        Create a detailed, helpful article that includes:
        1. Clear step-by-step instructions
        2. Common troubleshooting steps
        3. Prerequisites and requirements
        4. Examples where applicable
        5. Contact information for escalation
        
        Format your response as:
        TITLE: [Descriptive title]
        CATEGORY: [Technical/Hardware/Access/Network/Email/General]
        TAGS: [comma-separated relevant tags]
        CONTENT: [Detailed article content with clear sections and steps]
        
        Make it professional and easy to follow for both technical and non-technical users.
        """
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return parse_ai_article_response(response.text)
        
    except Exception as e:
        print(f"AI article generation error: {e}")
    
    return None

def parse_ai_article_response(ai_response):
    """Parse AI response into article structure"""
    try:
        lines = ai_response.split('\n')
        article = {
            'title': '',
            'category': 'General',
            'tags': [],
            'content': '',
            'ai_generated': True,
            'author': 'AI Assistant',
            'status': 'Published'
        }
        
        current_section = None
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('TITLE:'):
                article['title'] = line.replace('TITLE:', '').strip()
            elif line.startswith('CATEGORY:'):
                article['category'] = line.replace('CATEGORY:', '').strip()
            elif line.startswith('TAGS:'):
                tags_str = line.replace('TAGS:', '').strip()
                article['tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            elif line.startswith('CONTENT:'):
                current_section = 'content'
            elif current_section == 'content':
                content_lines.append(line)
        
        article['content'] = '\n'.join(content_lines).strip()
        
        # Ensure we have at least basic content
        if not article['title']:
            article['title'] = "AI Generated Article"
        if not article['content']:
            article['content'] = ai_response  # Fallback to full response
            
        return article
        
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return None

def ai_search_knowledge_base(query, max_results=5):
    """AI-powered knowledge base search with ranking"""
    if not AI_AVAILABLE:
        return search_knowledge_base(query)  # Fallback to basic search
    
    try:
        knowledge_base = get_knowledge_base()
        
        if not knowledge_base:
            return []
        
        # Create article summaries for AI
        articles_summary = ""
        article_ids = []
        
        for article_id, article in knowledge_base.items():
            articles_summary += f"ID: {article_id}\n"
            articles_summary += f"Title: {article.get('title', 'No title')}\n"
            articles_summary += f"Category: {article.get('category', 'General')}\n"
            articles_summary += f"Tags: {', '.join(article.get('tags', []))}\n"
            articles_summary += f"Content Preview: {article.get('content', '')[:200]}...\n\n"
            article_ids.append(article_id)
        
        prompt = f"""
        Search Query: "{query}"
        
        Knowledge Base Articles:
        {articles_summary}
        
        Rank the most relevant articles for this query. Consider:
        1. Title relevance
        2. Content match
        3. Tag relevance
        4. Category appropriateness
        
        Return the top {max_results} most relevant article IDs, one per line.
        If no articles are relevant, return "NONE".
        Only return article IDs that exist in the list above.
        """
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            result_lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            
            if result_lines and result_lines[0] == "NONE":
                return []
            
            relevant_articles = []
            for line in result_lines[:max_results]:
                article_id = line.strip()
                if article_id in knowledge_base:
                    article = knowledge_base[article_id].copy()
                    article['article_id'] = article_id
                    relevant_articles.append(article)
            
            return relevant_articles
            
    except Exception as e:
        print(f"AI search error: {e}")
    
    # Fallback to basic search
    return search_knowledge_base(query)

def search_knowledge_base(query):
    """Basic knowledge base search"""
    knowledge_base = get_knowledge_base()
    matching_articles = []
    
    query_lower = query.lower()
    
    for article_id, article in knowledge_base.items():
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        tags = ' '.join(article.get('tags', [])).lower()
        category = article.get('category', '').lower()
        
        # Simple keyword matching
        if (query_lower in title or 
            query_lower in content or 
            query_lower in tags or 
            query_lower in category):
            
            article['article_id'] = article_id
            matching_articles.append(article)
    
    return matching_articles

def update_knowledge_article(article_id, updates):
    """Update knowledge base article"""
    updates['updated_at'] = int(time.time())
    
    for field, value in updates.items():
        url = f"{FIREBASE_DB_URL}/knowledge_base/{article_id}/{field}.json"
        requests.put(url, json=value)
    
    return True

def get_knowledge_article(article_id):
    """Get specific knowledge article"""
    url = f"{FIREBASE_DB_URL}/knowledge_base/{article_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def initialize_ai_knowledge_base():
    """Initialize knowledge base with AI-generated articles"""
    knowledge_base = get_knowledge_base()
    
    if len(knowledge_base) < 5:  # Only if we have few articles
        default_topics = [
            ("Password Reset Procedures", "Step-by-step guide for password resets in corporate environment"),
            ("VPN Connection Setup", "Dubai and Singapore office VPN configuration"),
            ("Email Configuration Guide", "Corporate email setup for mobile and desktop"),
            ("Printer Network Setup", "Office printer installation and troubleshooting"),
            ("Identity Management System", "Access and authentication procedures"),
            ("Network Troubleshooting", "Common network issues and solutions"),
            ("Software Installation Guide", "Standard software deployment procedures"),
            ("Security Best Practices", "Corporate security guidelines and policies")
        ]
        
        for topic, context in default_topics:
            if AI_AVAILABLE:
                article_data = ai_generate_knowledge_article(topic, context)
                if article_data:
                    create_knowledge_article(article_data)
                    time.sleep(1)  # Rate limiting

# ========== Firebase Authentication Functions ==========
def register_user(email, password):
    """Register user with Firebase Auth"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

def login_user(email, password):
    """Login user with Firebase Auth"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

def get_user_info(id_token):
    """Get user info from Firebase"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_WEB_API_KEY}"
    payload = {"idToken": id_token}
    response = requests.post(url, json=payload)
    return response.json()

# ========== Firebase Database Functions ==========
def save_user_profile(user_id, user_data):
    """Save user profile to Firebase"""
    url = f"{FIREBASE_DB_URL}/users/{user_id}.json"
    response = requests.put(url, json=user_data)
    return response.status_code == 200

def get_user_profile(user_id):
    """Get user profile from Firebase"""
    url = f"{FIREBASE_DB_URL}/users/{user_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_all_users():
    """Get all users from Firebase"""
    url = f"{FIREBASE_DB_URL}/users.json"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return response.json()
    return {}

def get_admin_users():
    """Get all admin users from Firebase"""
    users = get_all_users()
    admin_users = {}
    for user_id, user_data in users.items():
        if user_data.get('role') == 'Admin' and user_data.get('status') == 'Active':
            admin_users[user_id] = user_data
    return admin_users

def update_user_status(user_id, status):
    """Update user status (Active/Inactive)"""
    url = f"{FIREBASE_DB_URL}/users/{user_id}/status.json"
    response = requests.put(url, json=status)
    return response.status_code == 200

def update_admin_availability(user_id, availability):
    """Update admin availability status"""
    url = f"{FIREBASE_DB_URL}/users/{user_id}/availability.json"
    response = requests.put(url, json=availability)
    
    if response.status_code == 200:
        # Notify about availability change
        user_data = get_user_profile(user_id)
        if user_data:
            notify_system(f"Admin {user_data.get('name', 'Unknown')} is now {availability}")
    
    return response.status_code == 200

def create_ticket(ticket_data):
    """Create new ticket in Firebase"""
    ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    ticket_data['ticket_id'] = ticket_id
    ticket_data['created_at'] = int(time.time())
    ticket_data['last_updated'] = int(time.time())
    
    url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}.json"
    response = requests.put(url, json=ticket_data)
    
    if response.status_code == 200:
        # Send comprehensive notifications
        notify_all_admins(f"🎫 New ticket created: {ticket_data['title']} by {ticket_data['customer_name']}")
        notify_system(f"Ticket {ticket_id} created by {ticket_data['customer_name']}")
        
        # Log activity
        log_activity("ticket_created", ticket_data['customer_email'], f"Created ticket: {ticket_data['title']}")
        
        return ticket_id
    return None

def get_tickets(user_email=None, user_role=None):
    """Get tickets from Firebase"""
    url = f"{FIREBASE_DB_URL}/tickets.json"
    response = requests.get(url)
    
    if response.status_code == 200 and response.json():
        tickets = response.json()
        
        # Filter for customers to see only their tickets
        if user_role == "Customer" and user_email:
            filtered_tickets = {}
            for ticket_id, ticket_data in tickets.items():
                if ticket_data.get('customer_email') == user_email:
                    filtered_tickets[ticket_id] = ticket_data
            return filtered_tickets
        
        return tickets
    return {}

def update_ticket(ticket_id, updates):
    """Update ticket in Firebase"""
    updates['last_updated'] = int(time.time())
    
    # Get current ticket data for comparison
    current_ticket = get_ticket_details(ticket_id)
    
    # Update each field individually
    for field, value in updates.items():
        url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}/{field}.json"
        requests.put(url, json=value)
    
    # Send notifications for important changes
    if current_ticket:
        if 'status' in updates and updates['status'] != current_ticket.get('status'):
            notify_ticket_participants(ticket_id, f"🔄 Ticket status changed to: {updates['status']}")
            notify_system(f"Ticket {ticket_id} status changed to {updates['status']}")
        
        if 'assigned_to' in updates and updates['assigned_to'] != current_ticket.get('assigned_to'):
            if updates['assigned_to'] != 'Unassigned':
                notify_user_by_name(updates['assigned_to'], f"📋 Ticket {ticket_id} has been assigned to you")
            notify_system(f"Ticket {ticket_id} assigned to {updates['assigned_to']}")
        
        if 'priority' in updates and updates['priority'] != current_ticket.get('priority'):
            notify_ticket_participants(ticket_id, f"⚡ Ticket priority changed to: {updates['priority']}")
    
    return True

def get_ticket_details(ticket_id):
    """Get specific ticket details"""
    url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def assign_ticket(ticket_id, assigned_to):
    """Assign ticket to admin user"""
    url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}/assigned_to.json"
    response = requests.put(url, json=assigned_to)
    
    if response.status_code == 200:
        # Get ticket details for notification
        ticket_data = get_ticket_details(ticket_id)
        if ticket_data:
            notify_user_by_name(assigned_to, f"📋 Ticket {ticket_id} assigned: {ticket_data.get('title', 'Unknown')}")
            notify_ticket_participants(ticket_id, f"👤 Ticket assigned to {assigned_to}")
            log_activity("ticket_assigned", assigned_to, f"Assigned ticket {ticket_id}")
    
    return response.status_code == 200

def send_chat_message(ticket_id, sender_email, sender_name, sender_role, message):
    """Send chat message for ticket"""
    message_data = {
        'sender_email': sender_email,
        'sender_name': sender_name,
        'sender_role': sender_role,
        'message': message,
        'timestamp': int(time.time())
    }
    
    url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}/chat.json"
    response = requests.post(url, json=message_data)
    
    if response.status_code == 200:
        # Notify the other parties in the ticket
        ticket_data = get_ticket_details(ticket_id)
        if ticket_data:
            if sender_role == "Customer":
                # Notify assigned admin and all admins
                if ticket_data.get('assigned_to') and ticket_data.get('assigned_to') != 'Unassigned':
                    notify_user_by_name(ticket_data['assigned_to'], f"💬 New message in ticket {ticket_id} from {sender_name}")
                notify_all_admins(f"💬 New customer message in ticket {ticket_id}")
            else:
                # Notify customer
                customer_email = ticket_data.get('customer_email')
                if customer_email:
                    notify_user_by_email(customer_email, f"💬 New message in ticket {ticket_id} from Support")
    
    return response.status_code == 200

def get_chat_messages(ticket_id):
    """Get chat messages for ticket"""
    url = f"{FIREBASE_DB_URL}/tickets/{ticket_id}/chat.json"
    response = requests.get(url)
    
    if response.status_code == 200 and response.json():
        messages = response.json()
        # Convert to list and sort by timestamp
        message_list = []
        for msg_id, msg_data in messages.items():
            msg_data['id'] = msg_id
            message_list.append(msg_data)
        
        return sorted(message_list, key=lambda x: x.get('timestamp', 0))
    return []

# ========== Direct Chat System ==========
def create_direct_chat(customer_email, admin_email):
    """Create or get direct chat between customer and admin"""
    safe_customer = customer_email.replace('@', '_at_').replace('.', '_dot_')
    safe_admin = admin_email.replace('@', '_at_').replace('.', '_dot_')
    
    chat_id = f"chat_{safe_customer}_{safe_admin}"
    
    # Check if chat already exists
    url = f"{FIREBASE_DB_URL}/direct_chats/{chat_id}.json"
    response = requests.get(url)
    
    if response.status_code == 200 and not response.json():
        # Create new chat
        chat_data = {
            'chat_id': chat_id,
            'customer_email': customer_email,
            'admin_email': admin_email,
            'created_at': int(time.time()),
            'last_message_at': int(time.time()),
            'status': 'active'
        }
        requests.put(url, json=chat_data)
        
        # Notify admin about new chat request
        admin_data = get_user_profile(safe_admin)
        customer_data = get_user_profile(safe_customer)
        
        if admin_data and customer_data:
            notify_user_by_email(admin_email, f"💬 New chat request from {customer_data.get('name', customer_email)}")
    
    return chat_id

def send_direct_message(chat_id, sender_email, sender_name, sender_role, message):
    """Send direct message in chat"""
    message_data = {
        'sender_email': sender_email,
        'sender_name': sender_name,
        'sender_role': sender_role,
        'message': message,
        'timestamp': int(time.time())
    }
    
    # Add message to chat
    url = f"{FIREBASE_DB_URL}/direct_chats/{chat_id}/messages.json"
    response = requests.post(url, json=message_data)
    
    if response.status_code == 200:
        # Update last message timestamp
        timestamp_url = f"{FIREBASE_DB_URL}/direct_chats/{chat_id}/last_message_at.json"
        requests.put(timestamp_url, json=int(time.time()))
        
        # Get chat participants
        chat_data = get_direct_chat_details(chat_id)
        if chat_data:
            # Notify the other participant
            if sender_role == "Customer":
                admin_email = chat_data.get('admin_email')
                if admin_email:
                    notify_user_by_email(admin_email, f"💬 New direct message from {sender_name}")
            else:
                customer_email = chat_data.get('customer_email')
                if customer_email:
                    notify_user_by_email(customer_email, f"💬 New message from Support ({sender_name})")
    
    return response.status_code == 200

def get_direct_chat_details(chat_id):
    """Get direct chat details"""
    url = f"{FIREBASE_DB_URL}/direct_chats/{chat_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_direct_chat_messages(chat_id):
    """Get direct chat messages"""
    url = f"{FIREBASE_DB_URL}/direct_chats/{chat_id}/messages.json"
    response = requests.get(url)
    
    if response.status_code == 200 and response.json():
        messages = response.json()
        message_list = []
        for msg_id, msg_data in messages.items():
            msg_data['id'] = msg_id
            message_list.append(msg_data)
        
        return sorted(message_list, key=lambda x: x.get('timestamp', 0))
    return []

def get_user_chats(user_email, user_role):
    """Get all chats for a user"""
    url = f"{FIREBASE_DB_URL}/direct_chats.json"
    response = requests.get(url)
    
    user_chats = {}
    if response.status_code == 200 and response.json():
        all_chats = response.json()
        
        for chat_id, chat_data in all_chats.items():
            if user_role == "Customer" and chat_data.get('customer_email') == user_email:
                user_chats[chat_id] = chat_data
            elif user_role == "Admin" and chat_data.get('admin_email') == user_email:
                user_chats[chat_id] = chat_data
    
    return user_chats

# ========== Asset Management ==========
def create_asset(asset_data):
    """Create new asset in Firebase"""
    asset_id = str(uuid.uuid4())
    asset_data['asset_id'] = asset_id
    asset_data['created_at'] = int(time.time())
    
    url = f"{FIREBASE_DB_URL}/assets/{asset_id}.json"
    response = requests.put(url, json=asset_data)
    
    if response.status_code == 200:
        # Notify admins about new asset
        notify_all_admins(f"🏢 New asset added: {asset_data['name']} ({asset_data['serial_number']})")
        log_activity("asset_created", "system", f"Created asset: {asset_data['name']}")
    
    return response.status_code == 200

def get_assets():
    """Get all assets from Firebase"""
    url = f"{FIREBASE_DB_URL}/assets.json"
    response = requests.get(url)
    
    if response.status_code == 200 and response.json():
        return response.json()
    return {}

def update_asset(asset_id, updates):
    """Update asset in Firebase"""
    # Get current asset for comparison
    current_asset = get_asset_details(asset_id)
    
    for field, value in updates.items():
        url = f"{FIREBASE_DB_URL}/assets/{asset_id}/{field}.json"
        requests.put(url, json=value)
    
    # Notify about important changes
    if current_asset:
        if 'status' in updates and updates['status'] != current_asset.get('status'):
            notify_all_admins(f"🏢 Asset {current_asset.get('name', 'Unknown')} status changed to: {updates['status']}")
        
        if 'warranty_end' in updates:
            # Check if warranty is expiring soon
            try:
                warranty_date = datetime.strptime(updates['warranty_end'], '%Y-%m-%d')
                days_until_expiry = (warranty_date.date() - datetime.now().date()).days
                
                if days_until_expiry <= 30 and days_until_expiry > 0:
                    notify_all_admins(f"⚠️ Asset {current_asset.get('name', 'Unknown')} warranty expires in {days_until_expiry} days")
                elif days_until_expiry <= 0:
                    notify_all_admins(f"🚨 Asset {current_asset.get('name', 'Unknown')} warranty has expired")
            except:
                pass
    
    return True

def get_asset_details(asset_id):
    """Get specific asset details"""
    url = f"{FIREBASE_DB_URL}/assets/{asset_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# ========== Enhanced Notification System ==========
def notify_all_admins(message):
    """Send notification to all admin users"""
    notification_data = {
        'message': message,
        'timestamp': int(time.time()),
        'type': 'admin_broadcast',
        'read': False,
        'priority': 'normal'
    }
    
    url = f"{FIREBASE_DB_URL}/notifications/admin_broadcast.json"
    requests.post(url, json=notification_data)

def notify_user_by_email(user_email, message):
    """Send notification to specific user by email"""
    safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
    notification_data = {
        'message': message,
        'timestamp': int(time.time()),
        'read': False,
        'priority': 'normal'
    }
    
    url = f"{FIREBASE_DB_URL}/notifications/users/{safe_email}.json"
    requests.post(url, json=notification_data)

def notify_user_by_name(user_name, message):
    """Send notification to user by name (for admin assignment)"""
    # Find user by name
    users = get_all_users()
    for user_id, user_data in users.items():
        if user_data.get('name') == user_name:
            notify_user_by_email(user_data.get('email'), message)
            break

def notify_ticket_participants(ticket_id, message):
    """Notify all participants in a ticket"""
    ticket_data = get_ticket_details(ticket_id)
    if ticket_data:
        # Notify customer
        customer_email = ticket_data.get('customer_email')
        if customer_email:
            notify_user_by_email(customer_email, message)
        
        # Notify assigned admin
        assigned_to = ticket_data.get('assigned_to')
        if assigned_to and assigned_to != 'Unassigned':
            notify_user_by_name(assigned_to, message)

def notify_system(message):
    """Log system-wide notifications"""
    notification_data = {
        'message': message,
        'timestamp': int(time.time()),
        'type': 'system',
        'read': False
    }
    
    url = f"{FIREBASE_DB_URL}/notifications/system.json"
    requests.post(url, json=notification_data)

def get_admin_notifications():
    """Get notifications for admin users"""
    url = f"{FIREBASE_DB_URL}/notifications/admin_broadcast.json"
    response = requests.get(url)
    
    notifications = []
    if response.status_code == 200 and response.json():
        notifs = response.json()
        for notif_id, notif_data in notifs.items():
            notif_data['id'] = notif_id
            notifications.append(notif_data)
    
    return sorted(notifications, key=lambda x: x.get('timestamp', 0), reverse=True)

def get_user_notifications(user_email):
    """Get notifications for specific user"""
    safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
    url = f"{FIREBASE_DB_URL}/notifications/users/{safe_email}.json"
    response = requests.get(url)
    
    notifications = []
    if response.status_code == 200 and response.json():
        notifs = response.json()
        for notif_id, notif_data in notifs.items():
            notif_data['id'] = notif_id
            notifications.append(notif_data)
    
    return sorted(notifications, key=lambda x: x.get('timestamp', 0), reverse=True)

def mark_notification_read(notification_type, notification_id):
    """Mark notification as read"""
    if notification_type == 'admin':
        url = f"{FIREBASE_DB_URL}/notifications/admin_broadcast/{notification_id}/read.json"
    else:
        url = f"{FIREBASE_DB_URL}/notifications/users/{notification_type}/{notification_id}/read.json"
    
    requests.put(url, json=True)

# ========== Activity Logging ==========
def log_activity(action, user, details):
    """Log user activity"""
    activity_data = {
        'action': action,
        'user': user,
        'details': details,
        'timestamp': int(time.time()),
        'ip_address': 'streamlit_app'  # In real app, get actual IP
    }
    
    url = f"{FIREBASE_DB_URL}/activity_log.json"
    requests.post(url, json=activity_data)

def get_recent_activities(limit=20):
    """Get recent system activities"""
    url = f"{FIREBASE_DB_URL}/activity_log.json"
    response = requests.get(url)
    
    activities = []
    if response.status_code == 200 and response.json():
        acts = response.json()
        for act_id, act_data in acts.items():
            act_data['id'] = act_id
            activities.append(act_data)
    
    # Sort by timestamp and return latest
    sorted_activities = sorted(activities, key=lambda x: x.get('timestamp', 0), reverse=True)
    return sorted_activities[:limit]

# ========== Enhanced AI Functions with Knowledge Base ==========
def ai_analyze_ticket(title, description):
    """AI analyzes ticket and suggests category, priority, response with knowledge base"""
    if not AI_AVAILABLE:
        return "AI not available - manual analysis required"
    
    try:
        # Search knowledge base for relevant articles
        kb_articles = ai_search_knowledge_base(f"{title} {description}", max_results=3)
        kb_context = ""
        
        if kb_articles:
            kb_context = "\n\nRelevant Knowledge Base Articles:\n"
            for article in kb_articles:
                kb_context += f"- {article['title']}: {article['content'][:200]}...\n"
        
        prompt = f"""
        You are a helpdesk AI for GET Holdings (Identity Management & Dubai Operations).
        
        Analyze this ticket:
        Title: {title}
        Description: {description}
        
        {kb_context}
        
        Respond ONLY in this exact format:
        CATEGORY: [Technical/Hardware/Access/Network/Email/General]
        PRIORITY: [High/Medium/Low]
        SENTIMENT: [Positive/Neutral/Negative/Urgent]
        SUGGESTED_RESPONSE: [Professional 2-3 sentence response based on knowledge base if available]
        KNOWLEDGE_MATCH: [Yes/No - whether knowledge base has relevant information]
        ESCALATION_NEEDED: [Yes/No - whether this needs immediate escalation]
        """
        
        response = model.generate_content(prompt)
        return response.text if response and response.text else "AI analysis failed"
        
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"

def ai_suggest_response(ticket_info, user_message="", chat_context=""):
    """AI generates professional response for agents using knowledge base"""
    if not AI_AVAILABLE:
        return "AI not available - please write manual response"
    
    try:
        # Search knowledge base for relevant information
        search_query = f"{ticket_info} {user_message}"
        kb_articles = ai_search_knowledge_base(search_query, max_results=2)
        
        kb_context = ""
        if kb_articles:
            kb_context = "\n\nRelevant Knowledge Base Articles:\n"
            for article in kb_articles:
                kb_context += f"Title: {article['title']}\nContent: {article['content'][:400]}...\n\n"
        
        prompt = f"""
        You are a professional IT support agent at GET Holdings.
        
        Ticket Information: {ticket_info}
        Latest Customer Message: {user_message}
        Chat Context: {chat_context}
        
        {kb_context}
        
        Write a helpful, professional response (3-4 sentences) that:
        1. Acknowledges the customer's issue with empathy
        2. Uses information from knowledge base if relevant
        3. Provides clear, actionable next steps
        4. Sets realistic expectations with timeframes
        5. Maintains a friendly, professional tone
        
        If knowledge base has relevant information, reference it in your response.
        If the issue requires escalation, mention it appropriately.
        """
        
        response = model.generate_content(prompt)
        return response.text if response and response.text else "AI response generation failed"
        
    except Exception as e:
        return f"AI Error: {str(e)} - Please write manual response"

# ========== Enhanced PDF Generation with Multiple Engines ==========
def generate_ticket_pdf_weasyprint(ticket_data, chat_messages=None):
    """Generate PDF using WeasyPrint"""
    try:
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Support Ticket Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                .ticket-details { margin: 20px 0; }
                .detail-row { margin: 10px 0; }
                .label { font-weight: bold; display: inline-block; width: 150px; }
                .chat-section { margin-top: 30px; }
                .message { margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; }
                .admin-message { border-left-color: #28a745; }
                .customer-message { border-left-color: #007bff; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GET Holdings - Support Ticket</h1>
                <h2>{{ ticket_id }}</h2>
            </div>
            
            <div class="ticket-details">
                <div class="detail-row"><span class="label">Title:</span> {{ title }}</div>
                <div class="detail-row"><span class="label">Description:</span> {{ description }}</div>
                <div class="detail-row"><span class="label">Status:</span> {{ status }}</div>
                <div class="detail-row"><span class="label">Priority:</span> {{ priority }}</div>
                <div class="detail-row"><span class="label">Customer:</span> {{ customer_name }} ({{ customer_email }})</div>
                <div class="detail-row"><span class="label">Project:</span> {{ project }}</div>
                <div class="detail-row"><span class="label">Region:</span> {{ region }}</div>
                <div class="detail-row"><span class="label">Created:</span> {{ created_at }}</div>
                <div class="detail-row"><span class="label">Assigned To:</span> {{ assigned_to }}</div>
            </div>
            
            {% if chat_messages %}
            <div class="chat-section">
                <h3>Chat History</h3>
                {% for message in chat_messages %}
                <div class="message {{ 'admin-message' if message.sender_role == 'Admin' else 'customer-message' }}">
                    <strong>{{ message.sender_name }} ({{ message.sender_role }})</strong> - {{ message.timestamp }}<br>
                    {{ message.message }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        # Prepare data
        created_time = datetime.fromtimestamp(ticket_data.get('created_at', 0)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format chat messages
        formatted_messages = []
        if chat_messages:
            for msg in chat_messages[-10:]:  # Last 10 messages
                formatted_messages.append({
                    'sender_name': msg.get('sender_name', 'Unknown'),
                    'sender_role': msg.get('sender_role', 'Unknown'),
                    'message': msg.get('message', 'No message'),
                    'timestamp': datetime.fromtimestamp(msg.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M')
                })
        
        html_content = template.render(
            ticket_id=ticket_data.get('ticket_id', 'N/A'),
            title=ticket_data.get('title', 'N/A'),
            description=ticket_data.get('description', 'N/A'),
            status=ticket_data.get('status', 'N/A'),
            priority=ticket_data.get('priority', 'N/A'),
            customer_name=ticket_data.get('customer_name', 'N/A'),
            customer_email=ticket_data.get('customer_email', 'N/A'),
            project=ticket_data.get('project', 'N/A'),
            region=ticket_data.get('region', 'N/A'),
            created_at=created_time,
            assigned_to=ticket_data.get('assigned_to', 'Unassigned'),
            chat_messages=formatted_messages
        )
        
        # Generate PDF
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        return BytesIO(pdf_bytes)
        
    except Exception as e:
        st.error(f"WeasyPrint PDF generation error: {str(e)}")
        return None

def generate_ticket_pdf_matplotlib(ticket_data, chat_messages=None):
    """Generate PDF using Matplotlib"""
    try:
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, f"Support Ticket: {ticket_data.get('ticket_id', 'N/A')}", 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            
            # Ticket details
            y_pos = 0.85
            details = [
                ('Title', ticket_data.get('title', 'N/A')),
                ('Description', ticket_data.get('description', 'N/A')[:100] + '...'),
                ('Status', ticket_data.get('status', 'N/A')),
                ('Priority', ticket_data.get('priority', 'N/A')),
                ('Customer', f"{ticket_data.get('customer_name', 'N/A')} ({ticket_data.get('customer_email', 'N/A')})"),
                ('Project', ticket_data.get('project', 'N/A')),
                ('Created', datetime.fromtimestamp(ticket_data.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')),
                ('Assigned To', ticket_data.get('assigned_to', 'Unassigned'))
            ]
            
            for label, value in details:
                ax.text(0.1, y_pos, f"{label}:", fontweight='bold', transform=ax.transAxes)
                ax.text(0.3, y_pos, str(value), transform=ax.transAxes, wrap=True)
                y_pos -= 0.06
            
            # Chat messages
            if chat_messages:
                ax.text(0.1, y_pos - 0.03, "Recent Chat Messages:", fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.08
                
                for msg in chat_messages[-5:]:  # Last 5 messages
                    if y_pos < 0.1:
                        break
                    sender = msg.get('sender_name', 'Unknown')
                    role = msg.get('sender_role', 'Unknown')
                    message = msg.get('message', 'No message')[:50] + '...'
                    timestamp = datetime.fromtimestamp(msg.get('timestamp', 0)).strftime('%m/%d %H:%M')
                    
                    ax.text(0.1, y_pos, f"{sender} ({role}) - {timestamp}:", fontsize=8, fontweight='bold', transform=ax.transAxes)
                    ax.text(0.1, y_pos - 0.02, message, fontsize=8, transform=ax.transAxes)
                    y_pos -= 0.06
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Matplotlib PDF generation error: {str(e)}")
        return None

def generate_ticket_pdf(ticket_data, chat_messages=None):
    """Generate PDF using available engine"""
    if not PDF_AVAILABLE:
        return None
    
    if PDF_ENGINE == "weasyprint":
        return generate_ticket_pdf_weasyprint(ticket_data, chat_messages)
    elif PDF_ENGINE == "matplotlib":
        return generate_ticket_pdf_matplotlib(ticket_data, chat_messages)
    else:
        st.error("No PDF generation engine available")
        return None

def generate_asset_pdf(assets):
    """Generate asset PDF using available engine"""
    if not PDF_AVAILABLE:
        return None
    
    if PDF_ENGINE == "weasyprint":
        return generate_asset_pdf_weasyprint(assets)
    elif PDF_ENGINE == "matplotlib":
        return generate_asset_pdf_matplotlib(assets)
    else:
        return None

def generate_asset_pdf_weasyprint(assets):
    """Generate asset PDF using WeasyPrint"""
    try:
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Asset Inventory Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GET Holdings - Asset Inventory</h1>
                <p>Generated on {{ current_date }}</p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Asset Name</th>
                        <th>Serial Number</th>
                        <th>Location</th>
                        <th>Status</th>
                        <th>Warranty End</th>
                        <th>Assigned To</th>
                    </tr>
                </thead>
                <tbody>
                    {% for asset in assets %}
                    <tr>
                        <td>{{ asset.name }}</td>
                        <td>{{ asset.serial_number }}</td>
                        <td>{{ asset.location }}</td>
                        <td>{{ asset.status }}</td>
                        <td>{{ asset.warranty_end }}</td>
                        <td>{{ asset.assigned_to }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        # Prepare asset data
        asset_list = []
        for asset in assets.values():
            asset_list.append({
                'name': asset.get('name', 'N/A'),
                'serial_number': asset.get('serial_number', 'N/A'),
                'location': asset.get('location', 'N/A'),
                'status': asset.get('status', 'N/A'),
                'warranty_end': asset.get('warranty_end', 'N/A'),
                'assigned_to': asset.get('assigned_to', 'Unassigned')
            })
        
        html_content = template.render(
            current_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            assets=asset_list
        )
        
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        return BytesIO(pdf_bytes)
        
    except Exception as e:
        st.error(f"Asset PDF generation error: {str(e)}")
        return None

def generate_asset_pdf_matplotlib(assets):
    """Generate asset PDF using Matplotlib"""
    try:
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Landscape
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, "Asset Inventory Report", 
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            
            # Create table data
            table_data = [['Asset Name', 'Serial Number', 'Location', 'Status', 'Warranty End']]
            
            for asset in list(assets.values())[:20]:  # Limit to 20 assets per page
                table_data.append([
                    asset.get('name', 'N/A')[:20],
                    asset.get('serial_number', 'N/A'),
                    asset.get('location', 'N/A'),
                    asset.get('status', 'N/A'),
                    asset.get('warranty_end', 'N/A')
                ])
            
            # Create table
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='left', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Style header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Asset PDF generation error: {str(e)}")
        return None

# ========== Enhanced Chat UI with Dark Theme ==========
def render_dark_chat_styles():
    """Apply dark theme styles for chat"""
    st.markdown("""
    <style>
    /* Dark Chat Container */
    .dark-chat-container {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 500px;
        max-height: 700px;
        overflow-y: auto;
    }
    
    /* Chat Header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 15px 15px 0 0;
        font-weight: bold;
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Message Bubbles */
    .message-bubble {
        margin: 15px 0;
        display: flex;
        align-items: flex-end;
    }
    
    .message-bubble.sent {
        justify-content: flex-end;
    }
    
    .message-bubble.received {
        justify-content: flex-start;
    }
    
    .message-content {
        max-width: 70%;
        padding: 12px 18px;
        border-radius: 20px;
        position: relative;
        word-wrap: break-word;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Sent Messages (User) */
    .message-content.sent {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 5px;
        margin-left: 50px;
    }
    
    /* Received Messages (Support) */
    .message-content.received {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border-bottom-left-radius: 5px;
        margin-right: 50px;
        border-left: 4px solid #3498db;
    }
    
    /* Message Meta */
    .message-meta {
        font-size: 11px;
        opacity: 0.8;
        margin-top: 5px;
    }
    
    .message-meta.sent {
        text-align: right;
    }
    
    .message-meta.received {
        text-align: left;
        color: #3498db;
    }
    
    /* Sender Name */
    .sender-name {
        font-weight: bold;
        font-size: 13px;
        margin-bottom: 3px;
        display: block;
    }
    
    /* Message Input Styling */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        color: white !important;
        border: 2px solid rgba(52, 152, 219, 0.3) !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 10px rgba(52, 152, 219, 0.3) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* AI Button Special Styling */
    .ai-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    
    .ai-button:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%) !important;
    }
    
    /* Scrollbar Styling */
    .dark-chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .dark-chat-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .dark-chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .dark-chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Empty State */
    .empty-chat {
        text-align: center;
        color: rgba(255, 255, 255, 0.6);
        padding: 50px 20px;
        font-style: italic;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        color: rgba(255, 255, 255, 0.6);
        font-style: italic;
        margin: 10px 0;
    }
    
    /* Online Status */
    .online-status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .online-status.online {
        background-color: #2ecc71;
        box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
    }
    
    .online-status.busy {
        background-color: #f39c12;
        box-shadow: 0 0 10px rgba(243, 156, 18, 0.5);
    }
    
    .online-status.offline {
        background-color: #e74c3c;
        box-shadow: 0 0 10px rgba(231, 76, 60, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

def render_chat_message(message, is_sender, sender_name, timestamp, sender_role=""):
    """Render chat message with enhanced dark theme"""
    time_str = get_time_ago(timestamp)
    
    bubble_class = "sent" if is_sender else "received"
    align_class = "sent" if is_sender else "received"
    
    # Role-based styling
    role_color = "#3498db" if sender_role == "Admin" else "#e67e22"
    
    st.markdown(f"""
    <div class="message-bubble {bubble_class}">
        <div class="message-content {bubble_class}">
            <span class="sender-name" style="color: {'white' if is_sender else role_color};">
                {sender_name} {f'({sender_role})' if sender_role else ''}
            </span>
            <div class="message-text">
                {message}
            </div>
            <div class="message-meta {align_class}">
                {time_str}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== Utility Functions ==========
def format_timestamp(timestamp):
    """Format timestamp for display"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return 'Unknown'

def is_admin(user_email):
    """Check if user is admin"""
    user_profile = get_user_profile(user_email.replace('@', '_at_').replace('.', '_dot_'))
    if user_profile:
        return user_profile.get('role') == 'Admin'
    return False

def get_time_ago(timestamp):
    """Get human-readable time ago"""
    if not timestamp:
        return 'Unknown'
    
    now = datetime.now()
    msg_time = datetime.fromtimestamp(timestamp)
    diff = now - msg_time
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

# ========== Session State Initialization ==========
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'id_token' not in st.session_state:
    st.session_state.id_token = None
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = None
if 'chat_refresh' not in st.session_state:
    st.session_state.chat_refresh = 0
if 'knowledge_initialized' not in st.session_state:
    st.session_state.knowledge_initialized = False

# Initialize AI knowledge base on first run
if not st.session_state.knowledge_initialized and AI_AVAILABLE:
    initialize_ai_knowledge_base()
    st.session_state.knowledge_initialized = True

# ========== Authentication Pages ==========
def login_page():
    st.title("🔐 GET Holdings - Support Desk")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("✅ Firebase Connected")
    
    with col2:
        if AI_AVAILABLE:
            st.success(f"✅ AI: {working_model_name}")
        else:
            st.error("❌ AI Offline")
    
    with col3:
        if PDF_AVAILABLE:
            st.success(f"✅ PDF: {PDF_ENGINE}")
        else:
            st.error("❌ PDF: Install weasyprint")
    
    with col4:
        kb_articles = get_knowledge_base()
        st.info(f"📚 KB: {len(kb_articles)} articles")
    
    st.markdown("---")
    
    # Login/Register tabs
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to your account")
        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("🔐 Login", type="primary"):
                if email and password:
                    result = login_user(email, password)
                    
                    if "idToken" in result:
                        # Get user profile
                        safe_email = email.replace('@', '_at_').replace('.', '_dot_')
                        user_profile = get_user_profile(safe_email)
                        
                        if user_profile and user_profile.get('status') == 'Active':
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_profile
                            st.session_state.id_token = result["idToken"]
                            
                            # Update last login
                            update_url = f"{FIREBASE_DB_URL}/users/{safe_email}/last_login.json"
                            requests.put(update_url, json=int(time.time()))
                            
                            # Set admin availability to online
                            if user_profile.get('role') == 'Admin':
                                update_admin_availability(safe_email, 'Online')
                            
                            # Log login activity
                            log_activity("user_login", email, "User logged in")
                            
                            st.success("Login successful!")
                            st.rerun()
                        elif user_profile and user_profile.get('status') == 'Inactive':
                            st.error("Your account has been deactivated. Please contact administrator.")
                        else:
                            st.error("User profile not found or incomplete.")
                    else:
                        error_msg = result.get("error", {}).get("message", "Login failed")
                        st.error(f"Login failed: {error_msg}")
                else:
                    st.error("Please enter both email and password")
    
    with tab2:
        st.subheader("Create new account")
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_name = st.text_input("Full Name*")
                reg_email = st.text_input("Email Address*")
                reg_organization = st.text_input("Organization")
            
            with col2:
                reg_password = st.text_input("Password*", type="password")
                reg_role = st.selectbox("Account Type", ["Customer", "Admin"])
                
                # Admin registration code
                reg_admin_code = ""
                if reg_role == "Admin":
                    reg_admin_code = st.text_input("Admin Registration Code*", type="password")
            
            if st.form_submit_button("✅ Create Account", type="primary"):
                if reg_name and reg_email and reg_password:
                    # Validate admin code
                    if reg_role == "Admin" and reg_admin_code != ADMIN_REGISTRATION_CODE:
                        st.error("Invalid admin registration code")
                    else:
                        # Register with Firebase Auth
                        auth_result = register_user(reg_email, reg_password)
                        
                        if "idToken" in auth_result:
                            # Save user profile
                            safe_email = reg_email.replace('@', '_at_').replace('.', '_dot_')
                            user_profile = {
                                'name': reg_name,
                                'email': reg_email,
                                'role': reg_role,
                                'organization': reg_organization,
                                'status': 'Active',
                                'availability': 'Offline' if reg_role == 'Admin' else None,
                                'created_at': int(time.time()),
                                'last_login': None
                            }
                            
                            if save_user_profile(safe_email, user_profile):
                                # Send comprehensive notifications
                                if reg_role == "Customer":
                                    notify_all_admins(f"👤 New customer registered: {reg_name} ({reg_email}) from {reg_organization}")
                                    notify_system(f"Customer registration: {reg_name}")
                                else:
                                    notify_all_admins(f"👥 New admin registered: {reg_name} ({reg_email})")
                                    notify_system(f"Admin registration: {reg_name}")
                                
                                # Log registration activity
                                log_activity("user_registration", reg_email, f"New {reg_role} registered")
                                
                                st.success("Account created successfully! Please login.")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("Failed to save user profile")
                        else:
                            error_msg = auth_result.get("error", {}).get("message", "Registration failed")
                            st.error(f"Registration failed: {error_msg}")
                else:
                    st.error("Please fill in all required fields")
    
    # Demo accounts
    st.markdown("---")
    st.subheader("🧪 Demo Accounts")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Admin Account**\n📧 admin@getgroup.com\n🔑 admin123")
    
    with col2:
        st.info("**Customer Account**\n📧 customer@client.com\n🔑 customer123")
# ========== Missing Page Functions ==========

def asset_management_page():
    """Asset Management Page for Admins"""
    st.title("🏢 Asset Management")
    
    assets = get_assets()
    
    # Asset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assets", len(assets))
    
    with col2:
        active_assets = len([a for a in assets.values() if a.get('status') == 'Active'])
        st.metric("Active Assets", active_assets)
    
    with col3:
        # Assets with warranty expiring soon
        expiring_soon = 0
        for asset in assets.values():
            warranty_end = asset.get('warranty_end')
            if warranty_end:
                try:
                    warranty_date = datetime.strptime(warranty_end, '%Y-%m-%d')
                    days_until_expiry = (warranty_date.date() - datetime.now().date()).days
                    if 0 <= days_until_expiry <= 30:
                        expiring_soon += 1
                except:
                    pass
        st.metric("Warranty Expiring", expiring_soon)
    
    with col4:
        maintenance_due = len([a for a in assets.values() if a.get('status') == 'Maintenance'])
        st.metric("Maintenance Due", maintenance_due)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Asset List", "➕ Add Asset", "📊 Analytics"])
    
    with tab1:
        st.subheader("Asset Inventory")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            asset_types = list(set([a.get('type', 'Unknown') for a in assets.values()]))
            type_filter = st.selectbox("Asset Type", ["All"] + asset_types)
        
        with col2:
            locations = list(set([a.get('location', 'Unknown') for a in assets.values()]))
            location_filter = st.selectbox("Location", ["All"] + locations)
        
        with col3:
            statuses = list(set([a.get('status', 'Unknown') for a in assets.values()]))
            status_filter = st.selectbox("Status", ["All"] + statuses)
        
        # Apply filters
        filtered_assets = assets
        if type_filter != "All":
            filtered_assets = {aid: adata for aid, adata in filtered_assets.items() if adata.get('type') == type_filter}
        if location_filter != "All":
            filtered_assets = {aid: adata for aid, adata in filtered_assets.items() if adata.get('location') == location_filter}
        if status_filter != "All":
            filtered_assets = {aid: adata for aid, adata in filtered_assets.items() if adata.get('status') == status_filter}
        
        # Display assets
        for asset_id, asset in filtered_assets.items():
            status_color = {"Active": "🟢", "Inactive": "🔴", "Maintenance": "🟡", "Retired": "⚫"}.get(asset.get('status'), "⚪")
            
            with st.expander(f"{status_color} {asset.get('name', 'Unknown')} | {asset.get('serial_number', 'Unknown')} | {asset.get('location', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Name:** {asset.get('name', 'Unknown')}")
                    st.write(f"**Type:** {asset.get('type', 'Unknown')}")
                    st.write(f"**Serial Number:** {asset.get('serial_number', 'Unknown')}")
                    st.write(f"**Description:** {asset.get('description', 'No description')}")
                    st.write(f"**Location:** {asset.get('location', 'Unknown')}")
                    st.write(f"**Assigned To:** {asset.get('assigned_to', 'Unassigned')}")
                
                with col2:
                    st.write(f"**Status:** {asset.get('status', 'Unknown')}")
                    st.write(f"**Purchase Date:** {asset.get('purchase_date', 'Unknown')}")
                    st.write(f"**Warranty End:** {asset.get('warranty_end', 'Unknown')}")
                    st.write(f"**Value:** ${asset.get('value', '0')}")
                    
                    # Quick actions
                    new_status = st.selectbox("Update Status", 
                                            ["Active", "Inactive", "Maintenance", "Retired"],
                                            index=["Active", "Inactive", "Maintenance", "Retired"].index(asset.get('status', 'Active')),
                                            key=f"status_{asset_id}")
                    
                    if st.button("Update", key=f"update_{asset_id}"):
                        if update_asset(asset_id, {'status': new_status}):
                            st.success("Asset updated!")
                            st.rerun()
    
    with tab2:
        st.subheader("Add New Asset")
        
        with st.form("add_asset_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                asset_name = st.text_input("Asset Name*")
                asset_type = st.selectbox("Asset Type", ["Computer", "Server", "Network Equipment", "Printer", "Mobile Device", "Other"])
                serial_number = st.text_input("Serial Number*")
                location = st.selectbox("Location", ["Dubai Office", "Singapore Office", "Remote", "Warehouse", "Data Center"])
            
            with col2:
                purchase_date = st.date_input("Purchase Date")
                warranty_end = st.date_input("Warranty End Date")
                value = st.number_input("Value ($)", min_value=0.0, step=0.01)
                assigned_to = st.text_input("Assigned To")
            
            description = st.text_area("Description")
            
            if st.form_submit_button("➕ Add Asset"):
                if asset_name and serial_number:
                    asset_data = {
                        'name': asset_name,
                        'type': asset_type,
                        'serial_number': serial_number,
                        'description': description,
                        'location': location,
                        'status': 'Active',
                        'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                        'warranty_end': warranty_end.strftime('%Y-%m-%d'),
                        'value': value,
                        'assigned_to': assigned_to
                    }
                    
                    if create_asset(asset_data):
                        st.success("Asset added successfully!")
                        log_activity("asset_created", st.session_state.user_data['email'], f"Added asset: {asset_name}")
                        st.rerun()
                    else:
                        st.error("Failed to add asset")
                else:
                    st.error("Please fill in required fields")
    
    with tab3:
        st.subheader("Asset Analytics")
        
        if assets:
            col1, col2 = st.columns(2)
            
            with col1:
                # Asset distribution by type
                type_counts = {}
                for asset in assets.values():
                    asset_type = asset.get('type', 'Unknown')
                    type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
                
                if type_counts:
                    fig = px.pie(values=list(type_counts.values()), 
                               names=list(type_counts.keys()),
                               title="Assets by Type")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Asset distribution by location
                location_counts = {}
                for asset in assets.values():
                    location = asset.get('location', 'Unknown')
                    location_counts[location] = location_counts.get(location, 0) + 1
                
                if location_counts:
                    fig = px.bar(x=list(location_counts.keys()), 
                               y=list(location_counts.values()),
                               title="Assets by Location")
                    st.plotly_chart(fig, use_container_width=True)

def asset_lookup_page():
    """Asset Lookup Page for Customers"""
    st.title("🔍 Asset Lookup")
    
    st.subheader("Search Company Assets")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search by asset name, serial number, or location...")
    
    with col2:
        search_button = st.button("🔍 Search")
    
    if search_query and (search_button or len(search_query) > 2):
        assets = get_assets()
        matching_assets = []
        
        query_lower = search_query.lower()
        
        for asset_id, asset in assets.items():
            name = asset.get('name', '').lower()
            serial = asset.get('serial_number', '').lower()
            location = asset.get('location', '').lower()
            asset_type = asset.get('type', '').lower()
            
            if (query_lower in name or 
                query_lower in serial or 
                query_lower in location or 
                query_lower in asset_type):
                
                asset['asset_id'] = asset_id
                matching_assets.append(asset)
        
        if matching_assets:
            st.success(f"Found {len(matching_assets)} matching asset(s)")
            
            for asset in matching_assets:
                status_color = {"Active": "🟢", "Inactive": "🔴", "Maintenance": "🟡", "Retired": "⚫"}.get(asset.get('status'), "⚪")
                
                with st.expander(f"{status_color} {asset.get('name', 'Unknown')} | {asset.get('location', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {asset.get('name', 'Unknown')}")
                        st.write(f"**Type:** {asset.get('type', 'Unknown')}")
                        st.write(f"**Serial Number:** {asset.get('serial_number', 'Unknown')}")
                        st.write(f"**Location:** {asset.get('location', 'Unknown')}")
                    
                    with col2:
                        st.write(f"**Status:** {asset.get('status', 'Unknown')}")
                        st.write(f"**Assigned To:** {asset.get('assigned_to', 'Unassigned')}")
                        st.write(f"**Warranty End:** {asset.get('warranty_end', 'Unknown')}")
        else:
            st.info(f"No assets found matching '{search_query}'")
    
    # Show asset categories for browsing
    if not search_query:
        st.subheader("Browse by Category")
        
        assets = get_assets()
        asset_types = list(set([a.get('type', 'Unknown') for a in assets.values()]))
        
        cols = st.columns(min(len(asset_types), 4))
        
        for i, asset_type in enumerate(asset_types):
            with cols[i % 4]:
                type_assets = [a for a in assets.values() if a.get('type') == asset_type]
                st.metric(f"📦 {asset_type}", len(type_assets))

def user_management_page():
    """User Management Page for Admins"""
    st.title("👥 User Management")
    
    users = get_all_users()
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users))
    
    with col2:
        admin_count = len([u for u in users.values() if u.get('role') == 'Admin'])
        st.metric("Administrators", admin_count)
    
    with col3:
        customer_count = len([u for u in users.values() if u.get('role') == 'Customer'])
        st.metric("Customers", customer_count)
    
    with col4:
        active_count = len([u for u in users.values() if u.get('status') == 'Active'])
        st.metric("Active Users", active_count)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["👤 User List", "📊 User Analytics", "⚙️ Admin Controls"])
    
    with tab1:
        st.subheader("All Users")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            role_filter = st.selectbox("Role", ["All", "Admin", "Customer"])
        
        with col2:
            status_filter = st.selectbox("Status", ["All", "Active", "Inactive"])
        
        with col3:
            if st.button("🔄 Refresh Users"):
                st.rerun()
        
        # Apply filters
        filtered_users = users
        if role_filter != "All":
            filtered_users = {uid: udata for uid, udata in filtered_users.items() if udata.get('role') == role_filter}
        if status_filter != "All":
            filtered_users = {uid: udata for uid, udata in filtered_users.items() if udata.get('status') == status_filter}
        
        # Display users
        for user_id, user_data in filtered_users.items():
            role_icon = "👤" if user_data.get('role') == 'Customer' else "👥"
            status_color = "🟢" if user_data.get('status') == 'Active' else "🔴"
            
            with st.expander(f"{role_icon} {status_color} {user_data.get('name', 'Unknown')} | {user_data.get('email', 'Unknown')} | {user_data.get('role', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Name:** {user_data.get('name', 'Unknown')}")
                    st.write(f"**Email:** {user_data.get('email', 'Unknown')}")
                    st.write(f"**Role:** {user_data.get('role', 'Unknown')}")
                    st.write(f"**Organization:** {user_data.get('organization', 'Not specified')}")
                    st.write(f"**Created:** {format_timestamp(user_data.get('created_at'))}")
                    st.write(f"**Last Login:** {format_timestamp(user_data.get('last_login'))}")
                
                with col2:
                    st.write(f"**Status:** {user_data.get('status', 'Unknown')}")
                    
                    if user_data.get('role') == 'Admin':
                        availability = user_data.get('availability', 'Offline')
                        st.write(f"**Availability:** {availability}")
                        
                        status_icon = {"Online": "🟢", "Busy": "🟡", "Offline": "🔴"}.get(availability, "🔴")
                        st.write(f"{status_icon} {availability}")
                    
                    # User controls
                    current_status = user_data.get('status', 'Active')
                    new_status = st.selectbox("Update Status", 
                                            ["Active", "Inactive"],
                                            index=["Active", "Inactive"].index(current_status),
                                            key=f"status_{user_id}")
                    
                    if st.button("Update Status", key=f"update_status_{user_id}"):
                        if update_user_status(user_id, new_status):
                            st.success("User status updated!")
                            if new_status == 'Inactive':
                                notify_user_by_email(user_data.get('email'), "Your account has been deactivated. Please contact administrator if you believe this is an error.")
                            log_activity("user_status_updated", st.session_state.user_data['email'], f"Updated {user_data.get('name')} status to {new_status}")
                            st.rerun()
    
    with tab2:
        st.subheader("User Analytics")
        
        if users:
            col1, col2 = st.columns(2)
            
            with col1:
                # User distribution by role
                role_counts = {"Admin": 0, "Customer": 0}
                for user_data in users.values():
                    role = user_data.get('role', 'Unknown')
                    if role in role_counts:
                        role_counts[role] += 1
                
                fig = px.pie(values=list(role_counts.values()), 
                           names=list(role_counts.keys()),
                           title="Users by Role",
                           color_discrete_map={'Admin': '#e74c3c', 'Customer': '#3498db'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Registration over time (last 30 days)
                from datetime import timedelta
                
                daily_registrations = {}
                today = datetime.now().date()
                
                for i in range(30):
                    date = today - timedelta(days=i)
                    daily_registrations[date.strftime('%m-%d')] = 0
                
                for user_data in users.values():
                    created_at = user_data.get('created_at')
                    if created_at:
                        created_date = datetime.fromtimestamp(created_at).date()
                        if (today - created_date).days <= 30:
                            date_key = created_date.strftime('%m-%d')
                            if date_key in daily_registrations:
                                daily_registrations[date_key] += 1
                
                fig = px.line(x=list(daily_registrations.keys()), 
                            y=list(daily_registrations.values()),
                            title="User Registrations (Last 30 Days)",
                            markers=True)
                fig.update_layout(xaxis_title="Date", yaxis_title="New Users")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Administrative Controls")
        
        # Admin availability overview
        st.write("### 👥 Admin Team Status")
        
        admin_users = get_admin_users()
        
        if admin_users:
            for admin_id, admin_data in admin_users.items():
                name = admin_data.get('name', 'Unknown')
                availability = admin_data.get('availability', 'Offline')
                status_icon = {"Online": "🟢", "Busy": "🟡", "Offline": "🔴"}.get(availability, "🔴")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"{status_icon} **{name}** - {availability}")
                
                with col2:
                    # Force status change for other admins (super admin feature)
                    if st.button(f"Set Offline", key=f"force_offline_{admin_id}"):
                        if update_admin_availability(admin_id, 'Offline'):
                            st.success(f"Set {name} to offline")
                            st.rerun()
        
        st.markdown("---")
        
        # System notifications
        st.write("### 📢 Send System Notification")
        
        with st.form("system_notification"):
            notification_message = st.text_area("Notification Message")
            notification_type = st.selectbox("Send to", ["All Users", "All Admins", "All Customers"])
            
            if st.form_submit_button("📢 Send Notification"):
                if notification_message:
                    if notification_type == "All Admins":
                        notify_all_admins(f"📢 System Notice: {notification_message}")
                    elif notification_type == "All Customers":
                        for user_data in users.values():
                            if user_data.get('role') == 'Customer':
                                notify_user_by_email(user_data.get('email'), f"📢 System Notice: {notification_message}")
                    else:  # All Users
                        notify_all_admins(f"📢 System Notice: {notification_message}")
                        for user_data in users.values():
                            notify_user_by_email(user_data.get('email'), f"📢 System Notice: {notification_message}")
                    
                    st.success("Notification sent!")
                    log_activity("system_notification", st.session_state.user_data['email'], f"Sent notification to {notification_type}")

def pdf_reports_page():
    """PDF Reports Page for Admins"""
    st.title("📄 PDF Reports")
    
    if not PDF_AVAILABLE:
        st.error("❌ PDF generation is not available. Please install weasyprint or matplotlib.")
        st.code("pip install weasyprint")
        return
    
    st.info(f"✅ PDF Engine: {PDF_ENGINE}")
    
    # Report options
    tab1, tab2, tab3 = st.tabs(["🎫 Ticket Reports", "🏢 Asset Reports", "👥 User Reports"])
    
    with tab1:
        st.subheader("Ticket Reports")
        
        tickets = get_tickets()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Ticket Summary Report")
            st.write(f"Total Tickets: {len(tickets)}")
            
            if tickets:
                status_counts = {}
                priority_counts = {}
                
                for ticket in tickets.values():
                    status = ticket.get('status', 'Unknown')
                    priority = ticket.get('priority', 'Unknown')
                    
                    status_counts[status] = status_counts.get(status, 0) + 1
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                st.write("**By Status:**")
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")
                
                st.write("**By Priority:**")
                for priority, count in priority_counts.items():
                    st.write(f"- {priority}: {count}")
            
            if st.button("📄 Generate Tickets Summary PDF"):
                # Create a summary report
                if tickets:
                    # Use the first ticket structure to create a summary report
                    summary_data = {
                        'ticket_id': 'SUMMARY_REPORT',
                        'title': 'Ticket Summary Report',
                        'description': f'Total tickets: {len(tickets)}\n\nStatus breakdown:\n' + 
                                     '\n'.join([f'- {k}: {v}' for k, v in status_counts.items()]) +
                                     f'\n\nPriority breakdown:\n' + 
                                     '\n'.join([f'- {k}: {v}' for k, v in priority_counts.items()]),
                        'status': 'Generated',
                        'priority': 'Report',
                        'customer_name': 'System',
                        'customer_email': 'system@getholdings.com',
                        'project': 'All Projects',
                        'region': 'All Regions',
                        'created_at': int(time.time()),
                        'assigned_to': 'Generated Report'
                    }
                    
                    pdf_buffer = generate_ticket_pdf(summary_data)
                    if pdf_buffer:
                        st.download_button(
                            label="⬇️ Download Summary PDF",
                            data=pdf_buffer,
                            file_name=f"tickets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
        
        with col2:
            st.write("### 🎫 Individual Ticket Reports")
            
            if tickets:
                ticket_options = []
                ticket_ids = []
                
                for ticket_id, ticket_data in tickets.items():
                    title = ticket_data.get('title', 'No title')
                    customer = ticket_data.get('customer_name', 'Unknown')
                    status = ticket_data.get('status', 'Unknown')
                    
                    ticket_options.append(f"{ticket_id} - {title} ({customer}) [{status}]")
                    ticket_ids.append(ticket_id)
                
                selected_ticket_idx = st.selectbox("Select Ticket", range(len(ticket_options)), 
                                                  format_func=lambda x: ticket_options[x])
                
                selected_ticket_id = ticket_ids[selected_ticket_idx]
                
                if st.button("📄 Generate Individual Ticket PDF"):
                    ticket_data = get_ticket_details(selected_ticket_id)
                    chat_messages = get_chat_messages(selected_ticket_id)
                    
                    if ticket_data:
                        pdf_buffer = generate_ticket_pdf(ticket_data, chat_messages)
                        if pdf_buffer:
                            st.download_button(
                                label="⬇️ Download Ticket PDF",
                                data=pdf_buffer,
                                file_name=f"ticket_{selected_ticket_id}.pdf",
                                mime="application/pdf"
                            )
    
    with tab2:
        st.subheader("Asset Reports")
        
        assets = get_assets()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🏢 Asset Inventory Report")
            st.write(f"Total Assets: {len(assets)}")
            
            if assets:
                type_counts = {}
                location_counts = {}
                status_counts = {}
                
                for asset in assets.values():
                    asset_type = asset.get('type', 'Unknown')
                    location = asset.get('location', 'Unknown')
                    status = asset.get('status', 'Unknown')
                    
                    type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
                    location_counts[location] = location_counts.get(location, 0) + 1
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                st.write("**By Type:**")
                for asset_type, count in type_counts.items():
                    st.write(f"- {asset_type}: {count}")
                
                st.write("**By Location:**")
                for location, count in location_counts.items():
                    st.write(f"- {location}: {count}")
            
            if st.button("📄 Generate Asset Inventory PDF"):
                if assets:
                    pdf_buffer = generate_asset_pdf(assets)
                    if pdf_buffer:
                        st.download_button(
                            label="⬇️ Download Asset Inventory PDF",
                            data=pdf_buffer,
                            file_name=f"asset_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
        
        with col2:
            st.write("### ⚠️ Warranty Expiration Report")
            
            expiring_assets = []
            
            for asset_id, asset in assets.items():
                warranty_end = asset.get('warranty_end')
                if warranty_end:
                    try:
                        warranty_date = datetime.strptime(warranty_end, '%Y-%m-%d')
                        days_until_expiry = (warranty_date.date() - datetime.now().date()).days
                        
                        if days_until_expiry <= 90:  # Expiring within 90 days
                            asset['asset_id'] = asset_id
                            asset['days_until_expiry'] = days_until_expiry
                            expiring_assets.append(asset)
                    except:
                        pass
            
            st.write(f"Assets with warranty expiring within 90 days: {len(expiring_assets)}")
            
            if expiring_assets:
                for asset in expiring_assets[:5]:  # Show first 5
                    days = asset['days_until_expiry']
                    if days < 0:
                        st.error(f"🚨 {asset.get('name')} - Expired {abs(days)} days ago")
                    elif days <= 30:
                        st.error(f"🔴 {asset.get('name')} - {days} days left")
                    else:
                        st.warning(f"🟡 {asset.get('name')} - {days} days left")
            
            if st.button("📄 Generate Warranty Report PDF"):
                # Create warranty report using asset PDF generator
                warranty_assets = {f"exp_{i}": asset for i, asset in enumerate(expiring_assets)}
                if warranty_assets:
                    pdf_buffer = generate_asset_pdf(warranty_assets)
                    if pdf_buffer:
                        st.download_button(
                            label="⬇️ Download Warranty Report PDF",
                            data=pdf_buffer,
                            file_name=f"warranty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
    
    with tab3:
        st.subheader("User Reports")
        
        users = get_all_users()
        
        st.write("### 👥 User Activity Report")
        st.write(f"Total Users: {len(users)}")
        
        if users:
            role_counts = {"Admin": 0, "Customer": 0}
            status_counts = {"Active": 0, "Inactive": 0}
            
            for user_data in users.values():
                role = user_data.get('role', 'Unknown')
                status = user_data.get('status', 'Unknown')
                
                if role in role_counts:
                    role_counts[role] += 1
                if status in status_counts:
                    status_counts[status] += 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**By Role:**")
                for role, count in role_counts.items():
                    st.write(f"- {role}: {count}")
            
            with col2:
                st.write("**By Status:**")
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")
            
            # Note: For user reports, we'd need to create a specific user PDF generator
            # This is left as a placeholder since the current PDF generators are for tickets and assets
            st.info("💡 User PDF reports can be implemented by extending the PDF generation functions")

def activity_log_page():
    """Activity Log Page for Admins"""
    st.title("📊 Activity Log")
    
    recent_activities = get_recent_activities(50)
    
    # Activity statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Activities", len(recent_activities))
    
    with col2:
        today = datetime.now().date()
        today_timestamp = int(time.mktime(today.timetuple()))
        today_activities = len([a for a in recent_activities if a.get('timestamp', 0) >= today_timestamp])
        st.metric("Today's Activities", today_activities)
    
    with col3:
        login_activities = len([a for a in recent_activities if a.get('action') == 'user_login'])
        st.metric("Logins", login_activities)
    
    with col4:
        ticket_activities = len([a for a in recent_activities if 'ticket' in a.get('action', '')])
        st.metric("Ticket Activities", ticket_activities)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_types = list(set([a.get('action', 'Unknown') for a in recent_activities]))
        action_filter = st.selectbox("Action Type", ["All"] + action_types)
    
    with col2:
        users_list = list(set([a.get('user', 'Unknown') for a in recent_activities]))
        user_filter = st.selectbox("User", ["All"] + users_list)
    
    with col3:
        if st.button("🔄 Refresh Log"):
            st.rerun()
    
    # Apply filters
    filtered_activities = recent_activities
    if action_filter != "All":
        filtered_activities = [a for a in filtered_activities if a.get('action') == action_filter]
    if user_filter != "All":
        filtered_activities = [a for a in filtered_activities if a.get('user') == user_filter]
    
    # Display activity log
    st.subheader(f"📋 Activity Log ({len(filtered_activities)} entries)")
    
    for activity in filtered_activities:
        action = activity.get('action', 'Unknown').replace('_', ' ').title()
        user = activity.get('user', 'Unknown')
        details = activity.get('details', 'No details')
        timestamp = format_timestamp(activity.get('timestamp'))
        
        # Action icon mapping
        action_icons = {
            'User Login': '🔐',
            'User Logout': '🚪',
            'User Registration': '📝',
            'Ticket Created': '🎫',
            'Ticket Updated': '✏️',
            'Ticket Assigned': '👤',
            'Knowledge Created': '📚',
            'Asset Created': '🏢',
            'System Notification': '📢'
        }
        
        icon = action_icons.get(action, '📌')
        
        with st.expander(f"{icon} {action} | {user} | {get_time_ago(activity.get('timestamp'))}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Action:** {action}")
                st.write(f"**User:** {user}")
                st.write(f"**Details:** {details}")
            
            with col2:
                st.write(f"**Timestamp:** {timestamp}")
                st.write(f"**IP:** {activity.get('ip_address', 'Unknown')}")
    
    # Activity analytics
    st.markdown("---")
    st.subheader("📈 Activity Analytics")
    
    if recent_activities:
        col1, col2 = st.columns(2)
        
        with col1:
            # Activity by type
            action_counts = {}
            for activity in recent_activities:
                action = activity.get('action', 'Unknown').replace('_', ' ').title()
                action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts:
                fig = px.bar(x=list(action_counts.keys()), 
                           y=list(action_counts.values()),
                           title="Activity by Type")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Activity over time (last 7 days)
            from datetime import timedelta
            
            daily_activity = {}
            today = datetime.now().date()
            
            for i in range(7):
                date = today - timedelta(days=i)
                daily_activity[date.strftime('%m-%d')] = 0
            
            for activity in recent_activities:
                timestamp = activity.get('timestamp')
                if timestamp:
                    activity_date = datetime.fromtimestamp(timestamp).date()
                    if (today - activity_date).days <= 7:
                        date_key = activity_date.strftime('%m-%d')
                        if date_key in daily_activity:
                            daily_activity[date_key] += 1
            
            fig = px.line(x=list(daily_activity.keys()), 
                        y=list(daily_activity.values()),
                        title="Activity Last 7 Days",
                        markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Activities")
            st.plotly_chart(fig, use_container_width=True)

def settings_page():
    """Settings Page for Users"""
    st.title("⚙️ Settings")
    
    user_data = st.session_state.user_data
    user_role = user_data.get('role', 'Customer')
    
    # Personal Settings
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🔔 Notifications", "🔧 System"])
    
    with tab1:
        st.subheader("Profile Settings")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                current_name = st.text_input("Full Name", value=user_data.get('name', ''))
                current_org = st.text_input("Organization", value=user_data.get('organization', ''))
            
            with col2:
                st.text_input("Email (Read-only)", value=user_data.get('email', ''), disabled=True)
                st.text_input("Role (Read-only)", value=user_data.get('role', ''), disabled=True)
            
            if st.form_submit_button("💾 Update Profile"):
                user_id = user_data['email'].replace('@', '_at_').replace('.', '_dot_')
                updates = {
                    'name': current_name,
                    'organization': current_org
                }
                
                # Update each field
                for field, value in updates.items():
                    url = f"{FIREBASE_DB_URL}/users/{user_id}/{field}.json"
                    requests.put(url, json=value)
                
                # Update session state
                st.session_state.user_data['name'] = current_name
                st.session_state.user_data['organization'] = current_org
                
                st.success("Profile updated successfully!")
                log_activity("profile_updated", user_data['email'], "Profile information updated")
                st.rerun()
    
    with tab2:
        st.subheader("Notification Settings")
        
        # Get current notifications
        if user_role == 'Admin':
            notifications = get_admin_notifications()
        else:
            notifications = get_user_notifications(user_data['email'])
        
        st.write(f"📧 You have {len(notifications)} notifications")
        
        # Display recent notifications
        if notifications:
            st.write("### Recent Notifications")
            
            for notif in notifications[:10]:
                read_status = "✅" if notif.get('read', False) else "🔔"
                time_ago = get_time_ago(notif.get('timestamp'))
                
                with st.expander(f"{read_status} {notif.get('message', 'No message')[:50]}... | {time_ago}"):
                    st.write(f"**Message:** {notif.get('message', 'No message')}")
                    st.write(f"**Time:** {format_timestamp(notif.get('timestamp'))}")
                    st.write(f"**Type:** {notif.get('type', 'Unknown')}")
                    
                    if not notif.get('read', False):
                        if st.button("Mark as Read", key=f"read_{notif.get('id')}"):
                            notification_type = 'admin' if user_role == 'Admin' else user_data['email'].replace('@', '_at_').replace('.', '_dot_')
                            mark_notification_read(notification_type, notif.get('id'))
                            st.success("Marked as read!")
                            st.rerun()
        
        # Notification preferences
        st.write("### 🔔 Notification Preferences")
        
        with st.form("notification_prefs"):
            email_notifications = st.checkbox("Email Notifications", value=True)
            ticket_updates = st.checkbox("Ticket Updates", value=True)
            system_announcements = st.checkbox("System Announcements", value=True)
            
            if user_role == 'Admin':
                new_tickets = st.checkbox("New Ticket Alerts", value=True)
                chat_messages = st.checkbox("Chat Message Alerts", value=True)
                user_registrations = st.checkbox("New User Registrations", value=True)
            
            if st.form_submit_button("💾 Save Preferences"):
                st.success("Notification preferences saved!")
                # In a real app, these would be saved to the database
    
    with tab3:
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔧 System Status")
            st.success("✅ Firebase Database Connected")
            
            if AI_AVAILABLE:
                st.success(f"✅ AI System: {working_model_name}")
            else:
                st.error("❌ AI System: Offline")
            
            if PDF_AVAILABLE:
                st.success(f"✅ PDF Generator: {PDF_ENGINE}")
            else:
                st.error("❌ PDF Generator: Not Available")
            
            # Knowledge base status
            kb_articles = get_knowledge_base()
            st.info(f"📚 Knowledge Base: {len(kb_articles)} articles")
        
        with col2:
            st.write("### 📊 Your Statistics")
            
            if user_role == 'Customer':
                # Customer stats
                user_tickets = get_tickets(user_email=user_data['email'], user_role='Customer')
                st.metric("My Tickets", len(user_tickets))
                
                open_tickets = len([t for t in user_tickets.values() if t.get('status') in ['Open', 'In Progress']])
                st.metric("Open Tickets", open_tickets)
                
                resolved_tickets = len([t for t in user_tickets.values() if t.get('status') == 'Resolved'])
                st.metric("Resolved Tickets", resolved_tickets)
            
            else:
                # Admin stats
                all_tickets = get_tickets()
                assigned_tickets = [t for t in all_tickets.values() if t.get('assigned_to') == user_data.get('name')]
                st.metric("Assigned Tickets", len(assigned_tickets))
                
                all_users = get_all_users()
                st.metric("Total Users", len(all_users))
                
                assets = get_assets()
                st.metric("Total Assets", len(assets))
        
        # Admin controls
        if user_role == 'Admin':
            st.markdown("---")
            st.write("### 🔧 Admin Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧹 Clear Old Notifications"):
                    st.info("This would clear notifications older than 30 days")
                
                if st.button("📊 Generate System Report"):
                    st.info("This would generate a comprehensive system report")
            
            with col2:
                if st.button("🔄 Refresh All Data"):
                    st.success("Data refreshed!")
                    st.rerun()
                
                if st.button("⚡ Test AI System"):
                    if AI_AVAILABLE:
                        with st.spinner("Testing AI..."):
                            try:
                                test_response = model.generate_content("Hello, please respond with 'AI system working correctly'")
                                if test_response and test_response.text:
                                    st.success(f"✅ AI Test Successful: {test_response.text}")
                                else:
                                    st.error("❌ AI Test Failed: No response")
                            except Exception as e:
                                st.error(f"❌ AI Test Failed: {str(e)}")
                    else:
                        st.error("❌ AI system not available")
        
        # Account controls
        st.markdown("---")
        st.write("### 🔐 Account Security")
        
        with st.form("security_form"):
            st.write("**Change Password** (Feature not implemented)")
            new_password = st.text_input("New Password", type="password", disabled=True)
            confirm_password = st.text_input("Confirm Password", type="password", disabled=True)
            
            if st.form_submit_button("🔐 Change Password", disabled=True):
                st.info("Password change functionality would be implemented with Firebase Auth")
        
        # Export data
        st.write("### 📥 Export Your Data")
        
        if st.button("📥 Export My Data"):
            if user_role == 'Customer':
                user_tickets = get_tickets(user_email=user_data['email'], user_role='Customer')
                export_data = {
                    'profile': user_data,
                    'tickets': user_tickets
                }
            else:
                # Admin export would include more data
                export_data = {
                    'profile': user_data,
                    'admin_stats': 'This would include admin-specific data'
                }
            
            st.json(export_data)
            st.info("💡 In a production app, this would generate a downloadable file")
# ========== Main Application with Enhanced Chat ==========
def main_app():
    user_data = st.session_state.user_data
    user_role = user_data.get('role', 'Customer')
    user_name = user_data.get('name', 'User')
    user_email = user_data.get('email', '')
    
    # Apply dark chat theme
    render_dark_chat_styles()
    
    # Sidebar
    st.sidebar.title("🎫 Support Desk")
    st.sidebar.markdown("---")
    
    # User info
    st.sidebar.write(f"👤 **{user_name}**")
    st.sidebar.write(f"📧 {user_email}")
    st.sidebar.write(f"🏷️ {user_role}")
    
    # Admin availability status
    if user_role == 'Admin':
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👥 Your Status")
        
        current_availability = user_data.get('availability', 'Offline')
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if current_availability == 'Online':
                st.success("🟢 Online")
            elif current_availability == 'Busy':
                st.warning("🟡 Busy")
            else:
                st.error("🔴 Offline")
        
        with col2:
            new_status = st.selectbox("Status", ["Online", "Busy", "Offline"], 
                                    index=["Online", "Busy", "Offline"].index(current_availability),
                                    key="admin_status")
        
        if new_status != current_availability:
            safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
            if update_admin_availability(safe_email, new_status):
                st.sidebar.success("Status updated!")
                st.session_state.user_data['availability'] = new_status
                st.rerun()
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success("✅ Firebase")
    
    if AI_AVAILABLE:
        st.sidebar.success(f"✅ AI: {working_model_name}")
    else:
        st.sidebar.error("❌ AI")
    
    if PDF_AVAILABLE:
        st.sidebar.success(f"✅ PDF: {PDF_ENGINE}")
    else:
        st.sidebar.error("❌ PDF")
    
    # Knowledge base status
    kb_articles = get_knowledge_base()
    st.sidebar.info(f"📚 KB: {len(kb_articles)} articles")
    
    # Navigation
    st.sidebar.markdown("---")
    if user_role == 'Admin':
        menu_options = ["Dashboard", "Tickets", "Live Chat Center", "AI Knowledge Base", "Asset Management", "User Management", "PDF Reports", "Activity Log", "Settings"]
    else:
        menu_options = ["My Tickets", "Submit Ticket", "Chat with Support", "Knowledge Base", "Asset Lookup"]
    
    selected_page = st.sidebar.selectbox("Navigate to:", menu_options)
    
    # Auto-refresh for real-time updates
    if st.sidebar.button("🔄 Refresh"):
        st.rerun()
    
    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        # Set admin status to offline
        if user_role == 'Admin':
            safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
            update_admin_availability(safe_email, 'Offline')
        
        # Log logout activity
        log_activity("user_logout", user_email, "User logged out")
        
        st.session_state.logged_in = False
        st.session_state.user_data = {}
        st.session_state.id_token = None
        st.session_state.selected_chat = None
        st.rerun()
    
    # Main content routing
    if selected_page == "Dashboard":
        dashboard_page()
    elif selected_page == "Tickets":
        tickets_page()
    elif selected_page == "My Tickets":
        my_tickets_page()
    elif selected_page == "Submit Ticket":
        submit_ticket_page()
    elif selected_page == "Live Chat Center":
        admin_chat_center()
    elif selected_page == "Chat with Support":
        customer_chat_page()
    elif selected_page == "Knowledge Base" or selected_page == "AI Knowledge Base":
        ai_knowledge_base_page()
    elif selected_page == "Asset Management":
        asset_management_page()
    elif selected_page == "Asset Lookup":
        asset_lookup_page()
    elif selected_page == "User Management":
        user_management_page()
    elif selected_page == "PDF Reports":
        pdf_reports_page()
    elif selected_page == "Activity Log":
        activity_log_page()
    elif selected_page == "Settings":
        settings_page()

# ========== Enhanced Knowledge Base Page ==========
def ai_knowledge_base_page():
    st.title("🤖 AI-Powered Knowledge Base")
    
    user_role = st.session_state.user_data.get('role', 'Customer')
    
    # AI Status and Stats
    col1, col2, col3, col4 = st.columns(4)
    
    knowledge_base = get_knowledge_base()
    ai_generated_count = len([a for a in knowledge_base.values() if a.get('ai_generated', False)])
    
    with col1:
        st.metric("Total Articles", len(knowledge_base))
    
    with col2:
        st.metric("AI Generated", ai_generated_count)
    
    with col3:
        categories = set([a.get('category', 'General') for a in knowledge_base.values()])
        st.metric("Categories", len(categories))
    
    with col4:
        if AI_AVAILABLE:
            st.success("🤖 AI Ready")
        else:
            st.error("🤖 AI Offline")
    
    # Enhanced Search with AI
    st.subheader("🔍 Intelligent Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("", placeholder="Ask anything about GET Holdings systems, procedures, or troubleshooting...")
    
    with col2:
        search_button = st.button("🔍 AI Search", type="primary")
    
    if search_query and (search_button or len(search_query) > 3):
        with st.spinner("🤖 AI searching knowledge base..."):
            if AI_AVAILABLE:
                matching_articles = ai_search_knowledge_base(search_query, max_results=5)
            else:
                matching_articles = search_knowledge_base(search_query)
        
        if matching_articles:
            st.success(f"🎯 Found {len(matching_articles)} relevant article(s)")
            
            for article in matching_articles:
                ai_badge = "🤖 AI Generated" if article.get('ai_generated', False) else "👤 Human Created"
                
                with st.expander(f"📄 {article['title']} | {article.get('category', 'General')} | {ai_badge}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Content:**")
                        st.write(article['content'])
                        
                        if article.get('tags'):
                            st.write(f"**Tags:** {', '.join(article['tags'])}")
                    
                    with col2:
                        st.write(f"**Category:** {article.get('category', 'General')}")
                        st.write(f"**Author:** {article.get('author', 'Unknown')}")
                        st.write(f"**Status:** {article.get('status', 'Published')}")
                        st.write(f"**Created:** {format_timestamp(article.get('created_at'))}")
                        
                        if article.get('ai_generated'):
                            st.info("🤖 This article was generated by AI")
                        
                        if user_role == 'Admin':
                            if st.button("✏️ Edit", key=f"edit_{article.get('article_id')}"):
                                st.session_state.edit_article_id = article.get('article_id')
                                st.rerun()
        else:
            st.info(f"🔍 No articles found for '{search_query}'")
            
            # AI can generate new article
            if AI_AVAILABLE and user_role == 'Admin':
                st.write("---")
                st.subheader("🤖 Generate New Knowledge Article")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Generate Article with AI", type="secondary"):
                        with st.spinner("🤖 AI generating article..."):
                            new_article = ai_generate_knowledge_article(search_query, "Based on user search query")
                            
                            if new_article:
                                if create_knowledge_article(new_article):
                                    st.success(f"✅ AI generated article: {new_article['title']}")
                                    st.rerun()
                                else:
                                    st.error("Failed to save AI-generated article")
                            else:
                                st.error("AI failed to generate article")
                
                with col2:
                    st.info("💡 **AI Article Generation**\nAI can create comprehensive articles based on your search query to help future users.")
    
    # Browse by Category
    if not search_query:
        st.subheader("📂 Browse by Category")
        
        # Group articles by category
        categories = {}
        for article_id, article in knowledge_base.items():
            category = article.get('category', 'General')
            if category not in categories:
                categories[category] = []
            article['article_id'] = article_id
            categories[category].append(article)
        
        # Display categories in columns
        if categories:
            category_names = list(categories.keys())
            cols = st.columns(min(len(category_names), 3))
            
            for i, (category, articles) in enumerate(categories.items()):
                with cols[i % 3]:
                    st.write(f"### 📁 {category}")
                    st.write(f"{len(articles)} articles")
                    
                    # Show recent articles
                    recent_articles = sorted(articles, key=lambda x: x.get('created_at', 0), reverse=True)[:3]
                    
                    for article in recent_articles:
                        ai_icon = "🤖" if article.get('ai_generated', False) else "👤"
                        if st.button(f"{ai_icon} {article['title'][:30]}...", key=f"cat_{article['article_id']}"):
                            # Show full article
                            with st.expander(f"📄 {article['title']}", expanded=True):
                                st.write(article['content'])
    
    # Admin Functions
    if user_role == 'Admin':
        st.markdown("---")
        
        tabs = st.tabs(["📝 Create Article", "🤖 AI Generator", "📊 Analytics"])
        
        with tabs[0]:
            st.subheader("Create New Article")
            
            with st.form("new_article_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_title = st.text_input("Title*")
                    new_category = st.selectbox("Category", ["Technical", "Hardware", "Access", "Network", "Email", "General"])
                    new_status = st.selectbox("Status", ["Published", "Draft", "Archived"])
                
                with col2:
                    new_tags = st.text_input("Tags (comma-separated)")
                    new_author = st.text_input("Author", value=st.session_state.user_data.get('name', ''))
                
                new_content = st.text_area("Content*", height=200)
                
                if st.form_submit_button("📝 Create Article"):
                    if new_title and new_content:
                        article_data = {
                            'title': new_title,
                            'content': new_content,
                            'category': new_category,
                            'tags': [tag.strip() for tag in new_tags.split(',') if tag.strip()],
                            'author': new_author,
                            'status': new_status,
                            'ai_generated': False
                        }
                        
                        if create_knowledge_article(article_data):
                            st.success("Article created successfully!")
                            log_activity("knowledge_created", st.session_state.user_data['email'], f"Created article: {new_title}")
                            st.rerun()
                        else:
                            st.error("Failed to create article")
                    else:
                        st.error("Please fill in title and content")
        
        with tabs[1]:
            st.subheader("🤖 AI Article Generator")
            
            if AI_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    ai_topic = st.text_input("Topic/Subject", placeholder="e.g., VPN Setup for Remote Workers")
                    ai_context = st.text_area("Additional Context", placeholder="Specific requirements, target audience, etc.")
                
                with col2:
                    ai_category = st.selectbox("Target Category", ["Technical", "Hardware", "Access", "Network", "Email", "General"])
                    ai_complexity = st.selectbox("Complexity Level", ["Beginner", "Intermediate", "Advanced"])
                
                if st.button("🤖 Generate Article with AI", type="primary"):
                    if ai_topic:
                        with st.spinner("🤖 AI creating comprehensive article..."):
                            enhanced_context = f"{ai_context}\nComplexity: {ai_complexity}\nCategory: {ai_category}"
                            new_article = ai_generate_knowledge_article(ai_topic, enhanced_context)
                            
                            if new_article:
                                new_article['category'] = ai_category  # Override category
                                
                                # Preview before saving
                                st.subheader("📄 AI Generated Article Preview")
                                
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**Title:** {new_article['title']}")
                                    st.write("**Content:**")
                                    st.write(new_article['content'])
                                
                                with col2:
                                    st.write(f"**Category:** {new_article['category']}")
                                    st.write(f"**Tags:** {', '.join(new_article.get('tags', []))}")
                                    st.write("**AI Generated:** Yes")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("✅ Save Article", type="primary"):
                                        if create_knowledge_article(new_article):
                                            st.success(f"✅ AI article saved: {new_article['title']}")
                                            log_activity("ai_article_generated", st.session_state.user_data['email'], f"AI generated: {new_article['title']}")
                                            st.rerun()
                                        else:
                                            st.error("Failed to save article")
                                
                                with col2:
                                    if st.button("❌ Discard"):
                                        st.rerun()
                            else:
                                st.error("AI failed to generate article. Please try again with different topic or context.")
                    else:
                        st.error("Please enter a topic")
            else:
                st.error("🤖 AI is not available. Please check configuration.")
        
        with tabs[2]:
            st.subheader("📊 Knowledge Base Analytics")
            
            if knowledge_base:
                # Category distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    category_counts = {}
                    for article in knowledge_base.values():
                        cat = article.get('category', 'General')
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                    
                    if category_counts:
                        fig = px.pie(values=list(category_counts.values()), 
                                   names=list(category_counts.keys()),
                                   title="Articles by Category")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # AI vs Human created
                    ai_count = len([a for a in knowledge_base.values() if a.get('ai_generated', False)])
                    human_count = len(knowledge_base) - ai_count
                    
                    fig = px.bar(x=['Human Created', 'AI Generated'], 
                               y=[human_count, ai_count],
                               title="Content Creation Source",
                               color=['Human Created', 'AI Generated'],
                               color_discrete_map={'Human Created': '#3498db', 'AI Generated': '#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recent activity
                st.subheader("📈 Recent Knowledge Base Activity")
                
                recent_articles = sorted(knowledge_base.values(), 
                                       key=lambda x: x.get('created_at', 0), reverse=True)[:5]
                
                for article in recent_articles:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        ai_badge = "🤖" if article.get('ai_generated', False) else "👤"
                        st.write(f"{ai_badge} **{article['title']}**")
                    
                    with col2:
                        st.write(f"📁 {article.get('category', 'General')}")
                    
                    with col3:
                        st.write(f"📅 {format_timestamp(article.get('created_at'))}")

# ========== Enhanced Direct Chat Display ==========
def display_direct_chat(chat_id, user_email, user_name, user_role):
    """Display enhanced dark-themed chat interface"""
    chat_data = get_direct_chat_details(chat_id)
    
    if not chat_data:
        st.error("Chat not found")
        return
    
    # Chat header with enhanced styling
    if user_role == 'Customer':
        admin_email = chat_data.get('admin_email', 'Unknown')
        admin_data = get_user_profile(admin_email.replace('@', '_at_').replace('.', '_dot_'))
        other_party = admin_data.get('name', 'Support Agent') if admin_data else 'Support Agent'
        availability = admin_data.get('availability', 'Offline') if admin_data else 'Offline'
        
        status_class = availability.lower()
        status_icon = {"Online": "🟢", "Busy": "🟡", "Offline": "🔴"}.get(availability, "🔴")
        
        st.markdown(f"""
        <div class="chat-header">
            <span class="online-status {status_class}"></span>
            {status_icon} Chat with {other_party} ({availability})
        </div>
        """, unsafe_allow_html=True)
    else:
        customer_email = chat_data.get('customer_email', 'Unknown')
        customer_data = get_user_profile(customer_email.replace('@', '_at_').replace('.', '_dot_'))
        other_party = customer_data.get('name', 'Customer') if customer_data else 'Customer'
        
        st.markdown(f"""
        <div class="chat-header">
            💬 Chat with {other_party}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat messages container with dark theme
    st.markdown('<div class="dark-chat-container">', unsafe_allow_html=True)
    
    messages = get_direct_chat_messages(chat_id)
    
    if messages:
        for msg in messages[-20:]:  # Show last 20 messages
            sender_name = msg.get('sender_name', 'Unknown')
            sender_role = msg.get('sender_role', 'Unknown')
            message = msg.get('message', 'No message')
            timestamp = msg.get('timestamp', 0)
            is_sender = msg.get('sender_email') == user_email
            
            render_chat_message(message, is_sender, sender_name, timestamp, sender_role)
    else:
        st.markdown("""
        <div class="empty-chat">
            💬 No messages yet. Start the conversation!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced message input
    st.markdown("---")
    
    with st.form(f"message_form_{chat_id}"):
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            new_message = st.text_input("", placeholder="Type your message...", key=f"msg_input_{chat_id}")
        
        with col2:
            send_button = st.form_submit_button("📤 Send")
        
        with col3:
            if AI_AVAILABLE and user_role == 'Admin':
                ai_button = st.form_submit_button("🤖 AI Help")
            else:
                ai_button = False
        
        if send_button and new_message:
            if send_direct_message(chat_id, user_email, user_name, user_role, new_message):
                st.success("Message sent!")
                st.rerun()
            else:
                st.error("Failed to send message")
        
        if ai_button and user_role == 'Admin':
            # Enhanced AI response with knowledge base context
            recent_msgs = messages[-5:] if messages else []
            context = "\n".join([f"{m.get('sender_name')}: {m.get('message')}" for m in recent_msgs])
            
            with st.spinner("🤖 AI analyzing conversation and searching knowledge base..."):
                ai_response = ai_suggest_response("Live chat support", new_message, context)
                
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    margin: 15px 0;
                    box-shadow: 0 5px 20px rgba(240, 147, 251, 0.3);
                '>
                    <strong>🤖 AI Suggested Response:</strong><br><br>
                    {ai_response}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Chat", key=f"refresh_{chat_id}"):
            st.rerun()
    
    with col2:
        if st.button("❌ Close Chat", key=f"close_{chat_id}"):
            st.session_state.selected_chat = None
            st.rerun()

# ========== Dashboard Page ==========
def dashboard_page():
    st.title("📊 Support Dashboard")
    
    # Get data
    tickets = get_tickets()
    assets = get_assets()
    users = get_all_users()
    recent_activities = get_recent_activities(10)
    knowledge_base = get_knowledge_base()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(tickets))
    
    with col2:
        open_tickets = len([t for t in tickets.values() if t.get('status') == 'Open'])
        st.metric("Open Tickets", open_tickets)
    
    with col3:
        high_priority = len([t for t in tickets.values() if t.get('priority') == 'High'])
        st.metric("High Priority", high_priority)
    
    with col4:
        st.metric("KB Articles", len(knowledge_base))
    
    # Real-time status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        admin_users = get_admin_users()
        online_admins = len([a for a in admin_users.values() if a.get('availability') == 'Online'])
        st.metric("Online Admins", online_admins)
    
    with col2:
        today = datetime.now().date()
        today_timestamp = int(time.mktime(today.timetuple()))
        today_tickets = len([t for t in tickets.values() if t.get('created_at', 0) >= today_timestamp])
        st.metric("Today's Tickets", today_tickets)
    
    with col3:
        unresolved = len([t for t in tickets.values() if t.get('status') not in ['Resolved', 'Closed']])
        st.metric("Unresolved", unresolved)
    
    with col4:
        ai_articles = len([a for a in knowledge_base.values() if a.get('ai_generated', False)])
        st.metric("AI Articles", ai_articles)
    
    # Charts
    if tickets:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tickets by Status")
            status_counts = {}
            for ticket in tickets.values():
                status = ticket.get('status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                fig = px.pie(values=list(status_counts.values()), names=list(status_counts.keys()))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Tickets by Priority")
            priority_counts = {}
            for ticket in tickets.values():
                priority = ticket.get('priority', 'Unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            if priority_counts:
                fig = px.bar(x=list(priority_counts.keys()), y=list(priority_counts.values()),
                           color=list(priority_counts.keys()),
                           color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity and Tickets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Activity")
        if recent_activities:
            for activity in recent_activities[:8]:
                st.write(f"**{activity.get('action', 'Unknown').replace('_', ' ').title()}**")
                st.write(f"👤 {activity.get('user', 'Unknown')} - {activity.get('details', 'No details')}")
                st.write(f"🕒 {format_timestamp(activity.get('timestamp'))}")
                st.markdown("---")
        else:
            st.info("No recent activity")
    
    with col2:
        st.subheader("Recent Tickets")
        recent_tickets = sorted(tickets.values(), key=lambda x: x.get('created_at', 0), reverse=True)[:8]
        
        for ticket in recent_tickets:
            with st.expander(f"🎫 {ticket.get('ticket_id', 'Unknown')} - {ticket.get('title', 'No title')}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Status:** {ticket.get('status', 'Unknown')}")
                    st.write(f"**Priority:** {ticket.get('priority', 'Unknown')}")
                    st.write(f"**Customer:** {ticket.get('customer_name', 'Unknown')}")
                with col_b:
                    st.write(f"**Project:** {ticket.get('project', 'Unknown')}")
                    st.write(f"**Created:** {format_timestamp(ticket.get('created_at'))}")
                    st.write(f"**Assigned:** {ticket.get('assigned_to', 'Unassigned')}")

# ========== Tickets Page ==========
# ========== FIXED BUTTON KEYS - Replace the affected functions ==========

# Replace the tickets_page function with this fixed version:
def tickets_page():
    st.title("🎫 All Tickets")
    
    tickets = get_tickets()
    users = get_all_users()
    admin_users = {uid: udata for uid, udata in users.items() if udata.get('role') == 'Admin'}
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        statuses = list(set([t.get('status', 'Unknown') for t in tickets.values()]))
        status_filter = st.selectbox("Status", ["All"] + statuses, key="tickets_status_filter")
    
    with col2:
        priorities = list(set([t.get('priority', 'Unknown') for t in tickets.values()]))
        priority_filter = st.selectbox("Priority", ["All"] + priorities, key="tickets_priority_filter")
    
    with col3:
        assigned_to_list = list(set([t.get('assigned_to', 'Unassigned') for t in tickets.values()]))
        assigned_filter = st.selectbox("Assigned to", ["All"] + assigned_to_list, key="tickets_assigned_filter")
    
    with col4:
        if st.button("🔄 Refresh", key="tickets_page_refresh"):
            st.rerun()
    
    # Apply filters
    filtered_tickets = tickets
    if status_filter != "All":
        filtered_tickets = {tid: tdata for tid, tdata in filtered_tickets.items() if tdata.get('status') == status_filter}
    if priority_filter != "All":
        filtered_tickets = {tid: tdata for tid, tdata in filtered_tickets.items() if tdata.get('priority') == priority_filter}
    if assigned_filter != "All":
        filtered_tickets = {tid: tdata for tid, tdata in filtered_tickets.items() if tdata.get('assigned_to') == assigned_filter}
    
    # Display tickets
    for ticket_id, ticket in filtered_tickets.items():
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(ticket.get('priority'), "⚪")
        status_color = {"Open": "🔴", "In Progress": "🟡", "Resolved": "🟢", "Closed": "⚫"}.get(ticket.get('status'), "⚪")
        
        with st.expander(f"{priority_color} {status_color} {ticket.get('ticket_id', 'Unknown')} - {ticket.get('title', 'No title')} | {ticket.get('customer_name', 'Unknown')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {ticket.get('description', 'No description')}")
                st.write(f"**Customer:** {ticket.get('customer_name', 'Unknown')} ({ticket.get('customer_email', 'Unknown')})")
                st.write(f"**Project:** {ticket.get('project', 'Unknown')} ({ticket.get('region', 'Unknown')})")
                st.write(f"**Category:** {ticket.get('category', 'Unknown')}")
                
                # Enhanced Chat section with dark theme
                st.write("---")
                render_dark_chat_styles()
                
                st.markdown('<div class="dark-chat-container" style="min-height: 200px; max-height: 300px;">', unsafe_allow_html=True)
                
                chat_messages = get_chat_messages(ticket_id)
                
                if chat_messages:
                    for msg in chat_messages[-3:]:  # Show last 3 messages
                        sender = msg.get('sender_name', 'Unknown')
                        role = msg.get('sender_role', 'Unknown')
                        message = msg.get('message', 'No message')
                        timestamp = msg.get('timestamp', 0)
                        is_sender = role == 'Admin'
                        
                        render_chat_message(message, is_sender, sender, timestamp, role)
                    
                    if len(chat_messages) > 3:
                        st.write(f"... and {len(chat_messages) - 3} more messages")
                else:
                    st.markdown('<div class="empty-chat">No messages yet</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add message
                new_message = st.text_input("Add Message", key=f"msg_{ticket_id}")
                col_send, col_ai = st.columns(2)
                
                with col_send:
                    if st.button("Send Message", key=f"send_{ticket_id}"):
                        if new_message:
                            user_data = st.session_state.user_data
                            if send_chat_message(ticket_id, user_data['email'], user_data['name'], user_data['role'], new_message):
                                st.success("Message sent!")
                                st.rerun()
                            else:
                                st.error("Failed to send message")
                
                with col_ai:
                    if AI_AVAILABLE:
                        if st.button("🤖 AI Suggest", key=f"ai_{ticket_id}"):
                            with st.spinner("AI generating response..."):
                                ticket_info = f"Title: {ticket.get('title')}\nDescription: {ticket.get('description')}"
                                ai_response = ai_suggest_response(ticket_info, new_message)
                                st.info(f"**AI Suggested Response:**\n\n{ai_response}")
            
            with col2:
                st.write(f"**Status:** {ticket.get('status', 'Unknown')}")
                st.write(f"**Priority:** {ticket.get('priority', 'Unknown')}")
                st.write(f"**Assigned:** {ticket.get('assigned_to', 'Unassigned')}")
                st.write(f"**Created:** {format_timestamp(ticket.get('created_at'))}")
                st.write(f"**Last Updated:** {get_time_ago(ticket.get('last_updated'))}")
                
                # Update controls
                new_status = st.selectbox("Update Status", 
                                        ["Open", "In Progress", "Resolved", "Closed"], 
                                        index=["Open", "In Progress", "Resolved", "Closed"].index(ticket.get('status', 'Open')),
                                        key=f"status_{ticket_id}")
                
                # Assignment
                admin_names = ["Unassigned"] + [udata.get('name', 'Unknown') for udata in admin_users.values()]
                current_assigned = ticket.get('assigned_to', 'Unassigned')
                if current_assigned not in admin_names:
                    admin_names.append(current_assigned)
                
                new_assigned = st.selectbox("Assign to", admin_names, 
                                          index=admin_names.index(current_assigned) if current_assigned in admin_names else 0,
                                          key=f"assign_{ticket_id}")
                
                # Priority update
                new_priority = st.selectbox("Priority", 
                                          ["Low", "Medium", "High"],
                                          index=["Low", "Medium", "High"].index(ticket.get('priority', 'Low')),
                                          key=f"priority_{ticket_id}")
                
                if st.button("Update Ticket", key=f"update_{ticket_id}"):
                    updates = {
                        'status': new_status,
                        'assigned_to': new_assigned,
                        'priority': new_priority
                    }
                    if update_ticket(ticket_id, updates):
                        st.success("Ticket updated!")
                        log_activity("ticket_updated", st.session_state.user_data['email'], f"Updated ticket {ticket_id}")
                        st.rerun()
                    else:
                        st.error("Failed to update ticket")
                
                # Generate PDF
                if PDF_AVAILABLE:
                    if st.button("📄 Generate PDF", key=f"pdf_{ticket_id}"):
                        chat_messages = get_chat_messages(ticket_id)
                        pdf_buffer = generate_ticket_pdf(ticket, chat_messages)
                        if pdf_buffer:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_buffer,
                                file_name=f"ticket_{ticket.get('ticket_id', 'unknown')}.pdf",
                                mime="application/pdf",
                                key=f"download_{ticket_id}"
                            )

# Replace the dashboard_page function with this fixed version:
def dashboard_page():
    st.title("📊 Support Dashboard")
    
    # Get data
    tickets = get_tickets()
    assets = get_assets()
    users = get_all_users()
    recent_activities = get_recent_activities(10)
    knowledge_base = get_knowledge_base()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(tickets))
    
    with col2:
        open_tickets = len([t for t in tickets.values() if t.get('status') == 'Open'])
        st.metric("Open Tickets", open_tickets)
    
    with col3:
        high_priority = len([t for t in tickets.values() if t.get('priority') == 'High'])
        st.metric("High Priority", high_priority)
    
    with col4:
        st.metric("KB Articles", len(knowledge_base))
    
    # Real-time status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        admin_users = get_admin_users()
        online_admins = len([a for a in admin_users.values() if a.get('availability') == 'Online'])
        st.metric("Online Admins", online_admins)
    
    with col2:
        today = datetime.now().date()
        today_timestamp = int(time.mktime(today.timetuple()))
        today_tickets = len([t for t in tickets.values() if t.get('created_at', 0) >= today_timestamp])
        st.metric("Today's Tickets", today_tickets)
    
    with col3:
        unresolved = len([t for t in tickets.values() if t.get('status') not in ['Resolved', 'Closed']])
        st.metric("Unresolved", unresolved)
    
    with col4:
        ai_articles = len([a for a in knowledge_base.values() if a.get('ai_generated', False)])
        st.metric("AI Articles", ai_articles)
    
    # Dashboard refresh button
    if st.button("🔄 Refresh Dashboard", key="dashboard_refresh"):
        st.rerun()
    
    # Charts
    if tickets:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tickets by Status")
            status_counts = {}
            for ticket in tickets.values():
                status = ticket.get('status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                fig = px.pie(values=list(status_counts.values()), names=list(status_counts.keys()))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Tickets by Priority")
            priority_counts = {}
            for ticket in tickets.values():
                priority = ticket.get('priority', 'Unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            if priority_counts:
                fig = px.bar(x=list(priority_counts.keys()), y=list(priority_counts.values()),
                           color=list(priority_counts.keys()),
                           color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity and Tickets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Activity")
        if recent_activities:
            for activity in recent_activities[:8]:
                st.write(f"**{activity.get('action', 'Unknown').replace('_', ' ').title()}**")
                st.write(f"👤 {activity.get('user', 'Unknown')} - {activity.get('details', 'No details')}")
                st.write(f"🕒 {format_timestamp(activity.get('timestamp'))}")
                st.markdown("---")
        else:
            st.info("No recent activity")
    
    with col2:
        st.subheader("Recent Tickets")
        recent_tickets = sorted(tickets.values(), key=lambda x: x.get('created_at', 0), reverse=True)[:8]
        
        for ticket in recent_tickets:
            with st.expander(f"🎫 {ticket.get('ticket_id', 'Unknown')} - {ticket.get('title', 'No title')}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Status:** {ticket.get('status', 'Unknown')}")
                    st.write(f"**Priority:** {ticket.get('priority', 'Unknown')}")
                    st.write(f"**Customer:** {ticket.get('customer_name', 'Unknown')}")
                with col_b:
                    st.write(f"**Project:** {ticket.get('project', 'Unknown')}")
                    st.write(f"**Created:** {format_timestamp(ticket.get('created_at'))}")
                    st.write(f"**Assigned:** {ticket.get('assigned_to', 'Unassigned')}")

# ========== IMPROVED GEMINI AI CONFIGURATION ==========
# ========== Submit Ticket Page ==========
def submit_ticket_page():
    st.title("📝 Submit New Ticket")
    
    user_data = st.session_state.user_data
    
    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Ticket Title*")
            category = st.selectbox("Category", ["Technical", "Hardware", "Access", "Network", "Email", "General"])
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        
        with col2:
            project = st.selectbox("Project", ["Dubai Operations", "Identity Management", "Project Alpha", "General"])
            region = st.selectbox("Region", ["UAE", "Singapore", "India", "KSA"])
        
        description = st.text_area("Description*", height=120)
        
        # Enhanced AI Analysis
        if AI_AVAILABLE:
            st.markdown("---")
            st.subheader("🤖 AI Ticket Analysis & Suggestions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("🤖 Analyze with AI", type="secondary"):
                    if title and description:
                        with st.spinner("AI analyzing ticket and searching knowledge base..."):
                            ai_result = ai_analyze_ticket(title, description)
                            if ai_result:
                                st.success("✅ AI Analysis Complete!")
                                st.info(ai_result)
                                
                                # Also search for relevant KB articles
                                kb_articles = ai_search_knowledge_base(f"{title} {description}", max_results=3)
                                if kb_articles:
                                    st.write("**📚 Relevant Knowledge Base Articles:**")
                                    for article in kb_articles:
                                        st.write(f"• **{article['title']}** - {article['content'][:100]}...")
                            else:
                                st.error("AI analysis failed")
                    else:
                        st.warning("Please enter title and description first")
            
            with col2:
                st.info("💡 **AI Analysis helps with:**\n- Automatic categorization\n- Priority suggestion\n- Similar issue detection\n- Solution recommendations")
        
        # File upload
        uploaded_file = st.file_uploader("Attach Files", type=['pdf', 'doc', 'docx', 'png', 'jpg', 'txt'])
        
        if st.form_submit_button("🎫 Submit Ticket", type="primary"):
            if title and description:
                ticket_data = {
                    'title': title,
                    'description': description,
                    'status': 'Open',
                    'priority': priority,
                    'project': project,
                    'region': region,
                    'customer_email': user_data['email'],
                    'customer_name': user_data['name'],
                    'category': category,
                    'assigned_to': 'Unassigned'
                }
                
                ticket_id = create_ticket(ticket_data)
                
                if ticket_id:
                    st.success(f"🎫 Ticket {ticket_id} created successfully!")
                    st.info("You can track your ticket progress and chat with our support team in the 'My Tickets' section.")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to create ticket")
            else:
                st.error("Please fill in all required fields")
# Add this function to your main.py file, preferably after the submit_ticket_page function

def my_tickets_page():
    """My Tickets Page for Customers"""
    st.title("🎫 My Tickets")
    
    user_data = st.session_state.user_data
    user_email = user_data.get('email', '')
    user_name = user_data.get('name', '')
    
    # Get customer's tickets
    my_tickets = get_tickets(user_email=user_email, user_role='Customer')
    
    if not my_tickets:
        st.info("You haven't submitted any tickets yet.")
        if st.button("📝 Submit Your First Ticket"):
            # This would navigate to submit ticket page
            st.rerun()
        return
    
    # Ticket statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(my_tickets))
    
    with col2:
        open_tickets = len([t for t in my_tickets.values() if t.get('status') in ['Open', 'In Progress']])
        st.metric("Open Tickets", open_tickets)
    
    with col3:
        resolved_tickets = len([t for t in my_tickets.values() if t.get('status') == 'Resolved'])
        st.metric("Resolved", resolved_tickets)
    
    with col4:
        high_priority = len([t for t in my_tickets.values() if t.get('priority') == 'High'])
        st.metric("High Priority", high_priority)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        statuses = list(set([t.get('status', 'Unknown') for t in my_tickets.values()]))
        status_filter = st.selectbox("Status", ["All"] + statuses, key="my_tickets_status_filter")
    
    with col2:
        priorities = list(set([t.get('priority', 'Unknown') for t in my_tickets.values()]))
        priority_filter = st.selectbox("Priority", ["All"] + priorities, key="my_tickets_priority_filter")
    
    with col3:
        if st.button("🔄 Refresh My Tickets", key="my_tickets_refresh"):
            st.rerun()
    
    # Apply filters
    filtered_tickets = my_tickets
    if status_filter != "All":
        filtered_tickets = {tid: tdata for tid, tdata in filtered_tickets.items() if tdata.get('status') == status_filter}
    if priority_filter != "All":
        filtered_tickets = {tid: tdata for tid, tdata in filtered_tickets.items() if tdata.get('priority') == priority_filter}
    
    # Display tickets with enhanced UI
    render_dark_chat_styles()
    
    for ticket_id, ticket in filtered_tickets.items():
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(ticket.get('priority'), "⚪")
        status_color = {"Open": "🔴", "In Progress": "🟡", "Resolved": "🟢", "Closed": "⚫"}.get(ticket.get('status'), "⚪")
        
        with st.expander(f"{priority_color} {status_color} {ticket.get('ticket_id', 'Unknown')} - {ticket.get('title', 'No title')} | {ticket.get('status', 'Unknown')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Title:** {ticket.get('title', 'Unknown')}")
                st.write(f"**Description:** {ticket.get('description', 'No description')}")
                st.write(f"**Project:** {ticket.get('project', 'Unknown')} ({ticket.get('region', 'Unknown')})")
                st.write(f"**Category:** {ticket.get('category', 'Unknown')}")
                
                # Enhanced Chat section
                st.write("---")
                st.write("**💬 Chat with Support:**")
                
                st.markdown('<div class="dark-chat-container" style="min-height: 250px; max-height: 400px;">', unsafe_allow_html=True)
                
                chat_messages = get_chat_messages(ticket_id)
                
                if chat_messages:
                    for msg in chat_messages:
                        sender_name = msg.get('sender_name', 'Unknown')
                        sender_role = msg.get('sender_role', 'Unknown')
                        message = msg.get('message', 'No message')
                        timestamp = msg.get('timestamp', 0)
                        is_sender = msg.get('sender_email') == user_email
                        
                        render_chat_message(message, is_sender, sender_name, timestamp, sender_role)
                else:
                    st.markdown('<div class="empty-chat">No messages yet. Start a conversation with support!</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Message input
                with st.form(f"customer_message_form_{ticket_id}"):
                    new_message = st.text_input("Type your message...", key=f"customer_msg_{ticket_id}")
                    
                    col_send, col_info = st.columns([1, 2])
                    
                    with col_send:
                        if st.form_submit_button("📤 Send Message"):
                            if new_message:
                                if send_chat_message(ticket_id, user_email, user_name, 'Customer', new_message):
                                    st.success("Message sent!")
                                    st.rerun()
                                else:
                                    st.error("Failed to send message")
                            else:
                                st.warning("Please enter a message")
                    
                    with col_info:
                        st.info("💡 Support will respond during business hours")
            
            with col2:
                st.write(f"**Status:** {ticket.get('status', 'Unknown')}")
                st.write(f"**Priority:** {ticket.get('priority', 'Unknown')}")
                st.write(f"**Assigned to:** {ticket.get('assigned_to', 'Unassigned')}")
                st.write(f"**Created:** {format_timestamp(ticket.get('created_at'))}")
                st.write(f"**Last Updated:** {get_time_ago(ticket.get('last_updated'))}")
                
                # Customer actions
                st.write("---")
                st.write("**Actions:**")
                
                if ticket.get('status') == 'Resolved':
                    if st.button("✅ Mark as Closed", key=f"close_{ticket_id}"):
                        if update_ticket(ticket_id, {'status': 'Closed'}):
                            st.success("Ticket marked as closed!")
                            st.rerun()
                
                if ticket.get('status') == 'Closed':
                    if st.button("🔄 Reopen Ticket", key=f"reopen_{ticket_id}"):
                        if update_ticket(ticket_id, {'status': 'Open'}):
                            st.success("Ticket reopened!")
                            st.rerun()
                
                # Rating system (placeholder)
                if ticket.get('status') in ['Resolved', 'Closed']:
                    st.write("**Rate this support:**")
                    rating = st.selectbox("Rating", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], key=f"rating_{ticket_id}")
                    
                    if st.button("Submit Rating", key=f"rate_{ticket_id}"):
                        # In a real app, this would save the rating
                        st.success("Thank you for your feedback!")
                
                # PDF download for customer
                if PDF_AVAILABLE:
                    if st.button("📄 Download PDF", key=f"customer_pdf_{ticket_id}"):
                        chat_messages = get_chat_messages(ticket_id)
                        pdf_buffer = generate_ticket_pdf(ticket, chat_messages)
                        if pdf_buffer:
                            st.download_button(
                                label="⬇️ Download Ticket PDF",
                                data=pdf_buffer,
                                file_name=f"my_ticket_{ticket.get('ticket_id', 'unknown')}.pdf",
                                mime="application/pdf",
                                key=f"customer_download_{ticket_id}"
                            )
    
    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Submit New Ticket", type="primary"):
            # This would navigate to submit ticket page
            st.info("Navigate to 'Submit Ticket' from the sidebar menu")
    
    with col2:
        if st.button("💬 Start Live Chat"):
            # This would navigate to chat page
            st.info("Navigate to 'Chat with Support' from the sidebar menu")
    
    with col3:
        if st.button("📚 Browse Knowledge Base"):
            # This would navigate to knowledge base
            st.info("Navigate to 'Knowledge Base' from the sidebar menu")
    
    # Helpful tips
    st.markdown("---")
    st.subheader("💡 Helpful Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📋 Ticket Status Guide:**
        - 🔴 **Open**: Ticket submitted, awaiting review
        - 🟡 **In Progress**: Support is working on your issue
        - 🟢 **Resolved**: Issue fixed, please verify
        - ⚫ **Closed**: Issue confirmed resolved
        """)
    
    with col2:
        st.info("""
        **🚨 Priority Levels:**
        - 🔴 **High**: Critical issues, system down
        - 🟡 **Medium**: Important but not urgent
        - 🟢 **Low**: Questions, minor issues
        """)
# ========== Chat System Pages ==========
def customer_chat_page():
    st.title("💬 Chat with Support")
    
    render_dark_chat_styles()
    
    user_email = st.session_state.user_data.get('email', '')
    user_name = st.session_state.user_data.get('name', '')
    
    # Get available admins
    admin_users = get_admin_users()
    
    if not admin_users:
        st.error("No support agents are currently available. Please create a ticket instead.")
        if st.button("📝 Create Ticket"):
            st.switch_page("Submit Ticket")
        return
    
    # Admin selection with enhanced UI
    st.subheader("👥 Available Support Agents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        admin_options = []
        admin_emails = []
        
        for admin_id, admin_data in admin_users.items():
            name = admin_data.get('name', 'Unknown')
            email = admin_data.get('email', 'Unknown')
            availability = admin_data.get('availability', 'Offline')
            
            status_icon = {"Online": "🟢", "Busy": "🟡", "Offline": "🔴"}.get(availability, "🔴")
            admin_options.append(f"{status_icon} {name} ({availability})")
            admin_emails.append(email)
        
        if admin_options:
            selected_admin_idx = st.selectbox("Select Support Agent", range(len(admin_options)), 
                                            format_func=lambda x: admin_options[x])
            
            selected_admin_email = admin_emails[selected_admin_idx]
            
            if st.button("💬 Start New Chat", type="primary"):
                chat_id = create_direct_chat(user_email, selected_admin_email)
                st.session_state.selected_chat = chat_id
                st.rerun()
    
    with col2:
        st.info("**💡 Chat Tips:**\n- Choose an online agent for faster response\n- Be specific about your issue\n- Busy agents may take longer to respond\n- Check knowledge base first for quick solutions")
    
    # Display existing chats
    user_chats = get_user_chats(user_email, 'Customer')
    
    if user_chats:
        st.markdown("---")
        st.subheader("💬 Your Chat Sessions")
        
        for chat_id, chat_data in user_chats.items():
            admin_email = chat_data.get('admin_email', 'Unknown')
            admin_data = get_user_profile(admin_email.replace('@', '_at_').replace('.', '_dot_'))
            admin_name = admin_data.get('name', 'Unknown') if admin_data else 'Unknown'
            admin_availability = admin_data.get('availability', 'Offline') if admin_data else 'Offline'
            
            last_message_time = get_time_ago(chat_data.get('last_message_at'))
            status_icon = {"Online": "🟢", "Busy": "🟡", "Offline": "🔴"}.get(admin_availability, "🔴")
            
            if st.button(f"{status_icon} Chat with {admin_name} (Last: {last_message_time})", key=f"chat_{chat_id}"):
                st.session_state.selected_chat = chat_id
                st.rerun()
    
    # Display selected chat
    if st.session_state.selected_chat:
        st.markdown("---")
        display_direct_chat(st.session_state.selected_chat, user_email, user_name, 'Customer')

def admin_chat_center():
    st.title("💬 Live Chat Center")
    
    render_dark_chat_styles()
    
    user_email = st.session_state.user_data.get('email', '')
    user_name = st.session_state.user_data.get('name', '')
    
    # Get admin's chats
    admin_chats = get_user_chats(user_email, 'Admin')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("💬 Active Chats")
        
        if admin_chats:
            for chat_id, chat_data in admin_chats.items():
                customer_email = chat_data.get('customer_email', 'Unknown')
                customer_data = get_user_profile(customer_email.replace('@', '_at_').replace('.', '_dot_'))
                customer_name = customer_data.get('name', 'Unknown') if customer_data else 'Unknown'
                
                last_message_time = get_time_ago(chat_data.get('last_message_at'))
                
                # Check for unread messages
                messages = get_direct_chat_messages(chat_id)
                latest_message = messages[-1] if messages else None
                
                if latest_message and latest_message.get('sender_email') != user_email:
                    chat_button_text = f"💬 {customer_name} 🔴 (New: {last_message_time})"
                else:
                    chat_button_text = f"💬 {customer_name} (Last: {last_message_time})"
                
                if st.button(chat_button_text, key=f"admin_chat_{chat_id}"):
                    st.session_state.selected_chat = chat_id
                    st.rerun()
        else:
            st.info("No active chats. Customers can start chats with you from their dashboard.")
    
    with col2:
        if st.session_state.selected_chat:
            display_direct_chat(st.session_state.selected_chat, user_email, user_name, 'Admin')
        else:
            st.info("Select a chat from the left panel to start messaging.")
            
            # Show enhanced stats
            st.subheader("📊 Chat Statistics")
            
            if admin_chats:
                total_chats = len(admin_chats)
                
                # Count recent activity
                recent_chats = 0
                for chat_data in admin_chats.values():
                    last_message = chat_data.get('last_message_at', 0)
                    if last_message > (time.time() - 86400):  # Last 24 hours
                        recent_chats += 1
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Chats", total_chats)
                with col_b:
                    st.metric("Active Today", recent_chats)

# Placeholder for other functions that would continue...
# These would include: asset_management_page, user_management_page, pdf_reports_page, activity_log_page, settings_page

# ========== Main Application Flow ==========
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()