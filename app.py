import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import json
import hashlib
import base64
from io import BytesIO
import numpy as np
import requests
from typing import Dict, Any, Optional, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Miva M&E Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration - using Streamlit secrets in cloud, fallback for local
if 'database' in st.secrets:
    DB_CONFIG = {
        "host": st.secrets["database"]["host"],
        "port": st.secrets["database"]["port"],
        "user": st.secrets["database"]["user"],
        "password": st.secrets["database"]["password"],
        "database": st.secrets["database"]["database"],
    }
    
    # Auth config from secrets
    AUTH_USERNAME = st.secrets["auth"]["admin_username"]
    AUTH_PASSWORD = st.secrets["auth"]["admin_password"]
else:
    # Fallback for local development
    DB_CONFIG = {
        "host": "16.170.143.253",
        "port": 5432,
        "user": "admin",
        "password": "password123",
        "database": "miva_ai_db",
    }
    AUTH_USERNAME = "miva_admin"
    AUTH_PASSWORD = "password123"

# Miva logo URL
MIVA_LOGO_URL = "https://i.imgur.com/azwWWQN.jpeg"

# Brand colors
BRAND_COLORS = {
    "primary": "#2563EB",
    "secondary": "#6B7280", 
    "white": "#FFFFFF",
    "accent": "#EF4444",
    "success": "#10B981",
    "warning": "#F59E0B"
}

# --- HELPER & STYLING FUNCTIONS ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_miva_logo():
    """Load Miva logo from URL and return base64 encoded string."""
    try:
        response = requests.get(MIVA_LOGO_URL, timeout=10)
        response.raise_for_status()
        img_base64 = base64.b64encode(response.content).decode()
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        st.warning(f"Could not load Miva logo: {e}")
        return None

def display_miva_logo(width: int = 150):
    """Display Miva logo in sidebar or main area."""
    logo_data = load_miva_logo()
    if logo_data:
        st.image(logo_data, width=width)
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: {BRAND_COLORS['primary']}; 
                       color: white; border-radius: 10px; margin: 10px 0;">
            <h2 style="margin: 0;">MIVA</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.8em;">Open University</p>
        </div>
        """, unsafe_allow_html=True)

def load_css():
    """Inject custom CSS for professional styling."""
    st.markdown(f"""
    <style>
    .main {{ padding: 1rem; }}
    .metric-card {{ background: {BRAND_COLORS['white']}; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; }}
    .metric-value {{ font-size: 2rem; font-weight: bold; color: {BRAND_COLORS['primary']}; margin: 0; }}
    .metric-label {{ font-size: 0.875rem; color: {BRAND_COLORS['secondary']}; margin-bottom: 0.5rem; }}
    .metric-delta {{ font-size: 0.875rem; margin-top: 0.25rem; }}
    .delta-positive {{ color: {BRAND_COLORS['success']}; }}
    .delta-negative {{ color: {BRAND_COLORS['accent']}; }}
    .header-title {{ color: {BRAND_COLORS['primary']}; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }}
    .subheader {{ color: {BRAND_COLORS['secondary']}; font-size: 1.125rem; margin-bottom: 1.5rem; }}
    .sidebar .sidebar-content {{ background: #F9FAFB; }}
    .stButton > button {{ background-color: {BRAND_COLORS['primary']}; color: white; border: none; border-radius: 0.375rem; font-weight: 500; transition: all 0.2s; }}
    .stButton > button:hover {{ background-color: #1D4ED8; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }}
    .conversation-bubble {{ border-radius: 1rem; padding: 0.75rem 1rem; margin: 0.25rem 0; max-width: 70%; }}
    .user-bubble {{ background: {BRAND_COLORS['primary']}; color: white; margin-left: auto; margin-right: 0; }}
    .bot-bubble {{ background: #F3F4F6; color: #1F2937; }}
    .system-bubble {{ background: {BRAND_COLORS['secondary']}; color: white; opacity: 0.8; font-style: italic; }}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION & DATABASE FUNCTIONS ---

# IMPROVED: Using a cached resource for efficient, persistent database connections.
@st.cache_resource
def init_connection():
    """Initialize and cache the database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception:
        return None

# IMPROVED: This function checks the status of the cached connection.
def test_database_connection():
    conn = init_connection()
    if conn is None or conn.closed != 0:
        # Clear the cache if connection is bad
        st.cache_resource.clear()
        conn = init_connection()

    if conn and conn.closed == 0:
        return True, "Connection successful"
    return False, "Could not establish connection"


def show_database_status():
    is_connected, message = test_database_connection()
    if is_connected:
        st.sidebar.success("🟢 Database Connected")
    else:
        st.sidebar.error(f"🔴 Database Error: {message}")
    return is_connected

@st.cache_data(ttl=300)
def run_query(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Execute SQL query using the cached connection and return a DataFrame."""
    try:
        conn = init_connection()
        if conn is None or conn.closed != 0:
             st.cache_resource.clear()
             conn = init_connection()
             if conn is None:
                 st.error("Database connection failed permanently.")
                 return pd.DataFrame()
        
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

# CORRECTED: Merged the two `check_password` functions into one.
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")
        selected_role = st.session_state.get("selected_role", "Viewer")
        
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            st.session_state["password_correct"] = True
            st.session_state["user_role"] = selected_role
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 3rem; border-radius: 1rem; 
                       box-shadow: 0 10px 25px rgba(0,0,0,0.1); margin-top: 2rem;">
            <div style="text-align: center; margin-bottom: 2rem;">
        """, unsafe_allow_html=True)
        
        display_miva_logo(200)
        
        st.markdown(f"""
                <h1 style="color: {BRAND_COLORS['primary']}; margin: 1rem 0 0.5rem 0;">Miva Open University</h1>
                <h3 style="color: {BRAND_COLORS['secondary']}; margin-bottom: 2rem;">M&E Dashboard</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.text_input("Username", key="username", placeholder="Enter username")
        st.text_input("Password", type="password", key="password", placeholder="Enter password")
        
        role_options = ["Viewer", "Analyst", "Owner"]
        st.selectbox("Select Role", role_options, key="selected_role", help="Choose your access level")
        
        role_descriptions = {
            "Viewer": "👁️ Read-only access to all dashboards",
            "Analyst": "📊 Read access + export capabilities",
            "Owner": "🔧 Full administrative access + settings management"
        }
        selected_role = st.session_state.get("selected_role", "Viewer")
        st.info(role_descriptions.get(selected_role, ""))
        
        st.button("Login", on_click=password_entered, use_container_width=True, type="primary")
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😞 Username or password incorrect")
        
        with st.expander("ℹ️ Login Information"):
            st.markdown(f"""
            **Demo Credentials:**
            - Username: `{AUTH_USERNAME}`
            - Password: `password123`
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    return False

# --- DATA & KPI FUNCTIONS ---

def compute_dates(period_option: str, custom_start: date = None, custom_end: date = None) -> Tuple[datetime, datetime]:
    """Compute start and end dates based on period selection."""
    now = datetime.now()
    if period_option == "Last 7 days":
        return now - timedelta(days=7), now
    if period_option == "Last 14 days":
        return now - timedelta(days=14), now
    if period_option == "Last 30 days":
        return now - timedelta(days=30), now
    if period_option == "Last 90 days":
        return now - timedelta(days=90), now
    if period_option == "Custom" and custom_start and custom_end:
        return datetime.combine(custom_start, datetime.min.time()), datetime.combine(custom_end, datetime.max.time())
    return now - timedelta(days=7), now # Default

def delta_str(delta: float, is_percentage: bool = False) -> str:
    """Format delta value with appropriate symbol."""
    if delta is None or np.isnan(delta) or delta == 0:
        return ""
    symbol = "%" if is_percentage else ""
    return f"{'+' if delta > 0 else ''}{delta:.1f}{symbol}"

def get_kpis(start_date: datetime, end_date: datetime, previous_start: datetime, previous_end: datetime) -> Dict[str, Any]:
    """Calculate all KPIs for the dashboard for two periods."""
    def query_period(s, e):
        return run_query("""
            WITH kpi_data AS (
                SELECT 
                    COUNT(DISTINCT cm.session_id) as active_sessions, COUNT(cm.id) as total_messages,
                    COUNT(DISTINCT COALESCE(cs.user_id, ccs.user_id)) as unique_users, AVG(cf.rating::numeric) as avg_rating,
                    COUNT(cf.id) as total_feedback, COUNT(CASE WHEN cf.rating::int <= 2 THEN 1 END) as negative_feedback,
                    COUNT(DISTINCT ov.id) as otp_created, COUNT(CASE WHEN ov.is_verified THEN 1 END) as otp_verified
                FROM chat_messages cm
                LEFT JOIN chat_sessions cs ON cm.session_id = cs.session_id
                LEFT JOIN conversation_sessions ccs ON cm.session_id = ccs.session_id
                LEFT JOIN chat_feedback cf ON cm.session_id = cf.session_id AND cf.created_at BETWEEN %(start)s AND %(end)s
                LEFT JOIN otp_verifications ov ON ov.created_at BETWEEN %(start)s AND %(end)s
                WHERE cm.timestamp BETWEEN %(start)s AND %(end)s
            )
            SELECT 
                active_sessions, total_messages, unique_users, COALESCE(avg_rating, 0) as csat,
                CASE WHEN total_feedback > 0 THEN (negative_feedback * 100.0 / total_feedback) ELSE 0 END as negative_pct,
                CASE WHEN otp_created > 0 THEN (otp_verified * 100.0 / otp_created) ELSE 0 END as otp_rate
            FROM kpi_data
        """, {'start': s, 'end': e})

    current_kpis = query_period(start_date, end_date)
    previous_kpis = query_period(previous_start, previous_end)

    default_kpis = {'active_sessions': 0, 'total_messages': 0, 'unique_users': 0, 'csat': 0, 'negative_pct': 0, 'otp_rate': 0}
    if current_kpis.empty:
        return {**default_kpis, 'sessions_delta': 0, 'messages_delta': 0, 'users_delta': 0, 'csat_delta': 0, 'negative_delta': 0, 'otp_delta': 0}

    current = current_kpis.iloc[0]
    previous = previous_kpis.iloc[0] if not previous_kpis.empty else pd.Series(default_kpis)
    
    deltas = {f'{k}_delta': float(current[k] - previous[k]) for k in default_kpis}
    return {**current.to_dict(), **deltas}

def display_kpis(kpis: Dict[str, Any]):
    """Display KPI metrics in a professional layout."""
    cols = st.columns(6)
    metrics = [
        ("Active Sessions", "active_sessions", False), ("Total Messages", "total_messages", False),
        ("Unique Users", "unique_users", False), ("CSAT Score", "csat", False, "{:.2f}/5.0"),
        ("Negative Feedback", "negative_pct", True, "{:.1f}%"), ("OTP Verification Rate", "otp_rate", True, "{:.1f}%")
    ]
    for i, (label, key, is_pct, *fmt) in enumerate(metrics):
        delta = kpis.get(f'{key}_delta', 0)
        value = kpis.get(key, 0)
        formatted_value = fmt[0].format(value) if fmt else f"{int(value):,}"
        delta_color = "normal" if delta == 0 or (key == 'negative_pct' and delta <= 0) else "inverse"
        cols[i].metric(label, formatted_value, delta=delta_str(delta, is_pct), delta_color=delta_color)

# --- PAGE RENDERING FUNCTIONS (NOW FULLY IMPLEMENTED) ---

def show_overview_page(start_date, end_date, previous_start, previous_end):
    st.markdown("## 📊 Dashboard Overview")
    display_kpis(get_kpis(start_date, end_date, previous_start, previous_end))
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 📈 Trends Analysis")
    col1, col2 = st.columns(2)
    with col1:
        daily_messages = run_query("SELECT DATE(timestamp) as date, COUNT(*) as message_count FROM chat_messages WHERE timestamp BETWEEN %(s)s AND %(e)s GROUP BY 1 ORDER BY 1", {'s': start_date, 'e': end_date})
        if not daily_messages.empty:
            fig = px.line(daily_messages, x='date', y='message_count', title="Daily Message Volume", color_discrete_sequence=[BRAND_COLORS['primary']])
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        daily_sessions = run_query("SELECT DATE(created_at) as date, COUNT(DISTINCT session_id) as session_count FROM chat_sessions WHERE created_at BETWEEN %(s)s AND %(e)s GROUP BY 1 ORDER BY 1", {'s': start_date, 'e': end_date})
        if not daily_sessions.empty:
            fig = px.line(daily_sessions, x='date', y='session_count', title="Daily New Sessions", color_discrete_sequence=[BRAND_COLORS['success']])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🕐 Activity Heatmap")
    activity_data = run_query("SELECT EXTRACT(hour FROM timestamp) as hour, EXTRACT(dow FROM timestamp) as day_of_week, COUNT(*) as message_count FROM chat_messages WHERE timestamp BETWEEN %(s)s AND %(e)s GROUP BY 1, 2", {'s': start_date, 'e': end_date})
    if not activity_data.empty:
        heatmap_data = activity_data.pivot(index='hour', columns='day_of_week', values='message_count').fillna(0)
        day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        heatmap_data.columns = [day_labels[int(col)] for col in heatmap_data.columns]
        fig_heatmap = px.imshow(heatmap_data, title="Message Activity by Hour and Day of Week", labels=dict(x="Day of Week", y="Hour of Day", color="Messages"), color_continuous_scale="Blues")
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_feedback_page(start_date, end_date, min_rating, max_rating, feedback_type, email_domain):
    st.markdown("## 💬 Feedback Analysis")
    filters = ["cf.created_at BETWEEN %(start_date)s AND %(end_date)s"]
    params = {'start_date': start_date, 'end_date': end_date}
    if min_rating != 1 or max_rating != 5:
        filters.append("cf.rating::int BETWEEN %(min_rating)s AND %(max_rating)s")
        params.update({'min_rating': min_rating, 'max_rating': max_rating})
    if feedback_type != "All":
        filters.append("cf.feedback_type = %(feedback_type)s")
        params['feedback_type'] = feedback_type
    if email_domain:
        filters.append("cf.email LIKE %(email_pattern)s")
        params['email_pattern'] = f"%@{email_domain}"
    
    filter_clause = " AND ".join(filters)
    
    col1, col2 = st.columns(2)
    with col1:
        rating_dist = run_query(f"SELECT rating::int, COUNT(*) as count FROM chat_feedback cf WHERE {filter_clause} GROUP BY 1 ORDER BY 1", params)
        if not rating_dist.empty:
            st.plotly_chart(px.bar(rating_dist, x='rating', y='count', title="Rating Distribution"), use_container_width=True)
    with col2:
        type_dist = run_query(f"SELECT feedback_type, COUNT(*) as count FROM chat_feedback cf WHERE {filter_clause} GROUP BY 1 ORDER BY 2 DESC", params)
        if not type_dist.empty:
            st.plotly_chart(px.pie(type_dist, values='count', names='feedback_type', title="Feedback Type Distribution"), use_container_width=True)

    st.markdown("### 📝 Recent Feedback")
    feedback_data = run_query(f"SELECT cf.created_at, cf.session_id, cf.email, cf.rating::int as rating, cf.feedback_type, cf.comment FROM chat_feedback cf WHERE {filter_clause} ORDER BY cf.created_at DESC LIMIT 100", params)
    if not feedback_data.empty:
        st.dataframe(feedback_data, use_container_width=True)
    else:
        st.info("No feedback entries match the current filters.")

def show_conversations_page(start_date, end_date):
    st.markdown("## 🗨️ Conversation Viewer")
    st.info("Select a recent session to view its transcript.")
    recent_sessions = run_query("SELECT session_id, MAX(timestamp) as last_message FROM chat_messages WHERE timestamp BETWEEN %(s)s AND %(e)s GROUP BY 1 ORDER BY 2 DESC LIMIT 100", {'s': start_date, 'e': end_date})
    if recent_sessions.empty:
        st.warning("No conversation sessions found for the selected period.")
        return
    
    session_map = {f"{row['session_id']} ({row['last_message']:%b %d, %H:%M})": row['session_id'] for _, row in recent_sessions.iterrows()}
    selected_option = st.selectbox("Select a Session", options=list(session_map.keys()))
    
    if selected_option:
        messages = run_query("SELECT role, content FROM chat_messages WHERE session_id = %(sid)s ORDER BY timestamp ASC", {'sid': session_map[selected_option]})
        st.markdown(f"### Transcript for Session `{session_map[selected_option]}`")
        for _, msg in messages.iterrows():
            st.markdown(f'<div class="conversation-bubble {msg["role"]}-bubble">{msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

def show_sessions_page(start_date, end_date, only_active):
    st.markdown("## 👥 Session Analytics")
    query = """
        WITH sd AS (
            SELECT cs.session_id, COALESCE(cs.user_id, ccs.user_id) as user_id, cs.created_at as start, MAX(cm.timestamp) as last_msg, COUNT(cm.id) as msg_count
            FROM chat_sessions cs
            LEFT JOIN conversation_sessions ccs ON cs.session_id = ccs.session_id
            LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
            WHERE cs.created_at BETWEEN %(s)s AND %(e)s GROUP BY 1, 2, 3
        ) SELECT session_id, user_id, start, last_msg, (last_msg - start) as duration, msg_count FROM sd
    """
    params = {'s': start_date, 'e': end_date}
    if only_active:
        query += " WHERE last_msg >= NOW() - INTERVAL '60 minutes'"
    query += " ORDER BY last_msg DESC LIMIT 500"
    
    sessions_df = run_query(query, params)
    if not sessions_df.empty:
        st.dataframe(sessions_df, use_container_width=True)
    else:
        st.warning("No session data found for the selected criteria.")

def show_otp_page(start_date, end_date):
    st.markdown("## 🔐 OTP Verification Monitor")
    otp_data = run_query("SELECT id, phone_number, email, is_verified, created_at FROM otp_verifications WHERE created_at BETWEEN %(s)s AND %(e)s ORDER BY created_at DESC", {'s': start_date, 'e': end_date})
    if otp_data.empty:
        st.warning("No OTP data found for the selected period.")
        return

    total, verified = len(otp_data), otp_data['is_verified'].sum()
    rate = (verified / total * 100) if total > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total OTPs Sent", f"{total:,}")
    col2.metric("Successful", f"{verified:,}")
    col3.metric("Verification Rate", f"{rate:.2f}%")

    st.markdown("### Recent OTP Attempts")
    st.dataframe(otp_data, use_container_width=True)

def show_reports_page(start_date, end_date):
    st.markdown("## 📄 Reports & Export")
    user_role = st.session_state.get('user_role', 'Viewer')
    
    if user_role == 'Viewer':
        st.warning("🔒 You have read-only access. Exporting data requires 'Analyst' or 'Owner' permissions.")
        return
        
    reports = {"Feedback": "chat_feedback", "Sessions": "chat_sessions", "Messages": "chat_messages", "OTP Logs": "otp_verifications"}
    selection = st.selectbox("Choose a report to generate:", options=reports.keys())
    
    if st.button(f"Generate '{selection}' Report"):
        table = reports[selection]
        time_col = 'timestamp' if table == 'chat_messages' else 'created_at'
        df = run_query(f"SELECT * FROM {table} WHERE {time_col} BETWEEN %(s)s AND %(e)s", {'s': start_date, 'e': end_date})
        
        if not df.empty:
            st.success(f"Report generated with {len(df)} rows.")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button( "📥 Download CSV", csv, f"{table}_{date.today()}.csv", 'text/csv')
        else:
            st.warning("No data found for this period.")

# --- MAIN APPLICATION ---

def main():
    """Main application function."""
    load_css()
    
    if not check_password():
        return
        
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        display_miva_logo(120)
    with header_col2:
        st.markdown(f"""
        <div style="padding-top: 1rem;">
            <h1 class="header-title" style="margin-bottom: 0.25rem;">Miva Open University</h1>
            <p class="subheader" style="margin-bottom: 0.5rem;">Monitoring & Evaluation Dashboard</p>
            <div style="background: {BRAND_COLORS['primary']}; color: white; padding: 0.25rem 0.75rem; 
                        border-radius: 0.375rem; font-size: 0.75rem; display: inline-block;">
                Role: {st.session_state.get('user_role', 'Viewer')} 
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0; border: 1px solid #E5E7EB;'>", unsafe_allow_html=True)
    
    with st.sidebar:
        display_miva_logo(180)
        st.markdown("---")
        db_connected = show_database_status()
        st.markdown("## 📊 Dashboard Filters")
        
        period_options = ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Custom"]
        period = st.selectbox("📅 Time Period", period_options)
        
        custom_start, custom_end = None, None
        if period == "Custom":
            col1, col2 = st.columns(2)
            custom_start = col1.date_input("Start Date", value=date.today() - timedelta(days=30))
            custom_end = col2.date_input("End Date", value=date.today())
        
        start_date, end_date = compute_dates(period, custom_start, custom_end)
        period_length = (end_date - start_date).days
        previous_start = start_date - timedelta(days=period_length)
        previous_end = start_date
        
        st.markdown("### 🔍 Advanced Filters")
        widget_filter = st.text_input("🎯 Widget ID", placeholder="Enter widget ID...")
        rating_range = st.slider("⭐ Rating Range", 1, 5, (1, 5))
        min_rating, max_rating = rating_range
        feedback_type_options = ["All", "positive", "negative", "neutral", "bug_report", "suggestion"]
        feedback_type = st.selectbox("📝 Feedback Type", feedback_type_options)
        email_domain = st.text_input("📧 Email Domain", placeholder="e.g., gmail.com")
        only_active = st.checkbox("🟢 Only Active Sessions", value=True)
        
        st.markdown("---")
        st.markdown(f"**Current Role:** {st.session_state.get('user_role', 'Viewer')}")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.sidebar.markdown("## 📱 Navigation")
    pages = ["📊 Overview", "💬 Feedback", "🗨️ Conversations", "👥 Sessions", "🔐 OTP Monitor", "📄 Reports"]
    selected_page = st.sidebar.radio("Select Page", pages, label_visibility="collapsed")
    
    if not db_connected:
        st.error("🔌 Database connection required. Please check your settings and refresh.")
        return
    
    try:
        if selected_page == "📊 Overview":
            show_overview_page(start_date, end_date, previous_start, previous_end)
        elif selected_page == "💬 Feedback":
            show_feedback_page(start_date, end_date, min_rating, max_rating, feedback_type, email_domain)
        elif selected_page == "🗨️ Conversations":
            show_conversations_page(start_date, end_date)
        elif selected_page == "👥 Sessions":
            show_sessions_page(start_date, end_date, only_active)
        elif selected_page == "🔐 OTP Monitor":
            show_otp_page(start_date, end_date)
        elif selected_page == "📄 Reports":
            show_reports_page(start_date, end_date)
    except Exception as e:
        st.error(f"An error occurred while rendering this page: {e}")
        if st.session_state.get('user_role') == 'Owner':
            with st.expander("🔧 Debug Information (Owner Only)"):
                st.exception(e)

if __name__ == "__main__":
    main()
