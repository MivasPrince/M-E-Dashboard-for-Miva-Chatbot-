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

# Logo loading function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_miva_logo():
    """Load Miva logo from URL and return base64 encoded string."""
    try:
        response = requests.get(MIVA_LOGO_URL, timeout=10)
        response.raise_for_status()
        
        # Convert to base64 for embedding
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
        # Fallback text logo
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: {BRAND_COLORS['primary']}; 
                       color: white; border-radius: 10px; margin: 10px 0;">
            <h2 style="margin: 0;">MIVA</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.8em;">Open University</p>
        </div>
        """, unsafe_allow_html=True)

# Custom CSS for professional styling
def load_css():
    st.markdown(f"""
    <style>
    .main {{
        padding: 1rem;
    }}
    
    .metric-card {{
        background: {BRAND_COLORS['white']};
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {BRAND_COLORS['primary']};
        margin: 0;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {BRAND_COLORS['secondary']};
        margin-bottom: 0.5rem;
    }}
    
    .metric-delta {{
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }}
    
    .delta-positive {{
        color: {BRAND_COLORS['success']};
    }}
    
    .delta-negative {{
        color: {BRAND_COLORS['accent']};
    }}
    
    .header-title {{
        color: {BRAND_COLORS['primary']};
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    
    .subheader {{
        color: {BRAND_COLORS['secondary']};
        font-size: 1.125rem;
        margin-bottom: 1.5rem;
    }}
    
    .sidebar .sidebar-content {{
        background: #F9FAFB;
    }}
    
    .stButton > button {{
        background-color: {BRAND_COLORS['primary']};
        color: white;
        border: none;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }}
    
    .feedback-card {{
        background: {BRAND_COLORS['white']};
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }}
    
    .rating-star {{
        color: #FCD34D;
    }}
    
    .conversation-bubble {{
        border-radius: 1rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        max-width: 70%;
    }}
    
    .user-bubble {{
        background: {BRAND_COLORS['primary']};
        color: white;
        margin-left: auto;
        margin-right: 0;
    }}
    
    .bot-bubble {{
        background: #F3F4F6;
        color: #1F2937;
    }}
    
    .system-bubble {{
        background: {BRAND_COLORS['secondary']};
        color: white;
        opacity: 0.8;
        font-style: italic;
    }}
    
    .alert-negative {{
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }}
    
    .alert-positive {{
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# Authentication functions
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")
        
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            st.session_state["password_correct"] = True
            st.session_state["user_role"] = "Owner"  # Set role for authenticated user
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form with Miva branding
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 3rem; border-radius: 1rem; 
                       box-shadow: 0 10px 25px rgba(0,0,0,0.1); margin-top: 5rem;">
            <div style="text-align: center; margin-bottom: 2rem;">
        """, unsafe_allow_html=True)
        
        # Display logo
        display_miva_logo(200)
        
        st.markdown(f"""
                <h1 style="color: {BRAND_COLORS['primary']}; margin: 1rem 0 0.5rem 0;">
                    Miva Open University
                </h1>
                <h3 style="color: {BRAND_COLORS['secondary']}; margin-bottom: 2rem;">
                    M&E Dashboard
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.text_input("Username", key="username", placeholder="Enter username")
        st.text_input("Password", type="password", key="password", placeholder="Enter password")
        st.button("Login", on_click=password_entered, use_container_width=True, type="primary")
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😞 Username or password incorrect")
        
        st.markdown("</div>", unsafe_allow_html=True)
    return False

# Database connection functions
@st.cache_resource
def init_connection():
    """Initialize database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# FIX: Modified run_query to use the cached connection for efficiency
@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_query(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame with proper connection handling."""
    try:
        conn = init_connection() # Use the cached connection
        if not conn:
            return pd.DataFrame()
        
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        # If connection is bad, clear the resource cache and rerun
        if 'connection' in str(e).lower():
            st.cache_resource.clear()
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

# Date helper functions
def compute_dates(period_option: str, custom_start: date = None, custom_end: date = None) -> Tuple[datetime, datetime]:
    """Compute start and end dates based on period selection."""
    now = datetime.now()
    
    if period_option == "Last 7 days":
        start = now - timedelta(days=7)
        end = now
    elif period_option == "Last 14 days":
        start = now - timedelta(days=14)
        end = now
    elif period_option == "Last 30 days":
        start = now - timedelta(days=30)
        end = now
    elif period_option == "Last 90 days":
        start = now - timedelta(days=90)
        end = now
    elif period_option == "Custom" and custom_start and custom_end:
        start = datetime.combine(custom_start, datetime.min.time())
        end = datetime.combine(custom_end, datetime.max.time())
    else:
        # Default to last 7 days
        start = now - timedelta(days=7)
        end = now
    
    return start, end

def delta_str(delta: float, is_percentage: bool = False) -> str:
    """Format delta value with appropriate symbol."""
    if delta is None or np.isnan(delta) or delta == 0:
        return ""
    
    symbol = "%" if is_percentage else ""
    prefix = "+" if delta > 0 else ""
    return f"{prefix}{delta:.1f}{symbol}"

# KPI calculation functions
def get_kpis(start_date: datetime, end_date: datetime, previous_start: datetime, previous_end: datetime) -> Dict[str, Any]:
    """Calculate all KPIs for the dashboard."""
    
    # Current period KPIs
    current_kpis = run_query("""
        WITH kpi_data AS (
            SELECT 
                COUNT(DISTINCT cm.session_id) as active_sessions,
                COUNT(cm.id) as total_messages,
                COUNT(DISTINCT COALESCE(cs.user_id, ccs.user_id)) as unique_users,
                AVG(cf.rating::numeric) as avg_rating,
                COUNT(cf.id) as total_feedback,
                COUNT(CASE WHEN cf.rating::int <= 2 THEN 1 END) as negative_feedback,
                COUNT(DISTINCT ov.id) as otp_created,
                COUNT(CASE WHEN ov.is_verified THEN 1 END) as otp_verified
            FROM chat_messages cm
            LEFT JOIN chat_sessions cs ON cm.session_id = cs.session_id
            LEFT JOIN conversation_sessions ccs ON cm.session_id = ccs.session_id
            LEFT JOIN chat_feedback cf ON cm.session_id = cf.session_id 
                AND cf.created_at BETWEEN %(start_date)s AND %(end_date)s
            LEFT JOIN otp_verifications ov ON ov.created_at BETWEEN %(start_date)s AND %(end_date)s
            WHERE cm.timestamp BETWEEN %(start_date)s AND %(end_date)s
        )
        SELECT 
            active_sessions,
            total_messages,
            unique_users,
            COALESCE(avg_rating, 0) as csat,
            CASE 
                WHEN total_feedback > 0 THEN (negative_feedback * 100.0 / total_feedback)
                ELSE 0 
            END as negative_pct,
            CASE 
                WHEN otp_created > 0 THEN (otp_verified * 100.0 / otp_created)
                ELSE 0 
            END as otp_rate
        FROM kpi_data
    """, {
        'start_date': start_date,
        'end_date': end_date
    })
    
    # Previous period KPIs for delta calculation
    previous_kpis = run_query("""
        WITH kpi_data AS (
            SELECT 
                COUNT(DISTINCT cm.session_id) as active_sessions,
                COUNT(cm.id) as total_messages,
                COUNT(DISTINCT COALESCE(cs.user_id, ccs.user_id)) as unique_users,
                AVG(cf.rating::numeric) as avg_rating,
                COUNT(cf.id) as total_feedback,
                COUNT(CASE WHEN cf.rating::int <= 2 THEN 1 END) as negative_feedback,
                COUNT(DISTINCT ov.id) as otp_created,
                COUNT(CASE WHEN ov.is_verified THEN 1 END) as otp_verified
            FROM chat_messages cm
            LEFT JOIN chat_sessions cs ON cm.session_id = cs.session_id
            LEFT JOIN conversation_sessions ccs ON cm.session_id = ccs.session_id
            LEFT JOIN chat_feedback cf ON cm.session_id = cf.session_id 
                AND cf.created_at BETWEEN %(prev_start)s AND %(prev_end)s
            LEFT JOIN otp_verifications ov ON ov.created_at BETWEEN %(prev_start)s AND %(prev_end)s
            WHERE cm.timestamp BETWEEN %(prev_start)s AND %(prev_end)s
        )
        SELECT 
            active_sessions,
            total_messages,
            unique_users,
            COALESCE(avg_rating, 0) as csat,
            CASE 
                WHEN total_feedback > 0 THEN (negative_feedback * 100.0 / total_feedback)
                ELSE 0 
            END as negative_pct,
            CASE 
                WHEN otp_created > 0 THEN (otp_verified * 100.0 / otp_created)
                ELSE 0 
            END as otp_rate
        FROM kpi_data
    """, {
        'prev_start': previous_start,
        'prev_end': previous_end
    })
    
    if current_kpis.empty:
        return {
            'active_sessions': 0, 'total_messages': 0, 'unique_users': 0,
            'csat': 0, 'negative_pct': 0, 'otp_rate': 0,
            'sessions_delta': 0, 'messages_delta': 0, 'users_delta': 0,
            'csat_delta': 0, 'negative_delta': 0, 'otp_delta': 0
        }
    
    current = current_kpis.iloc[0]
    
    # Calculate deltas
    deltas = {}
    if not previous_kpis.empty:
        previous = previous_kpis.iloc[0]
        deltas = {
            'sessions_delta': float(current['active_sessions'] - previous['active_sessions']),
            'messages_delta': float(current['total_messages'] - previous['total_messages']),
            'users_delta': float(current['unique_users'] - previous['unique_users']),
            'csat_delta': float(current['csat'] - previous['csat']),
            'negative_delta': float(current['negative_pct'] - previous['negative_pct']),
            'otp_delta': float(current['otp_rate'] - previous['otp_rate'])
        }
    else:
        deltas = {
            'sessions_delta': 0.0, 'messages_delta': 0.0, 'users_delta': 0.0,
            'csat_delta': 0.0, 'negative_delta': 0.0, 'otp_delta': 0.0
        }
    
    return {**current.to_dict(), **deltas}

def display_kpis(kpis: Dict[str, Any]):
    """Display KPI metrics in a professional layout."""
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        delta = kpis.get('sessions_delta', 0)
        st.metric(
            "Active Sessions",
            f"{int(kpis.get('active_sessions', 0)):,}",
            delta=delta_str(delta)
        )
    
    with col2:
        delta = kpis.get('messages_delta', 0)
        st.metric(
            "Total Messages",
            f"{int(kpis.get('total_messages', 0)):,}",
            delta=delta_str(delta)
        )
    
    with col3:
        delta = kpis.get('users_delta', 0)
        st.metric(
            "Unique Users",
            f"{int(kpis.get('unique_users', 0)):,}",
            delta=delta_str(delta)
        )
    
    with col4:
        delta = kpis.get('csat_delta', 0)
        st.metric(
            "CSAT Score",
            f"{kpis.get('csat', 0):.2f}/5.0",
            delta=delta_str(delta)
        )
    
    with col5:
        delta = kpis.get('negative_delta', 0)
        # For negative feedback, a decrease is good (normal), an increase is bad (inverse)
        delta_color = "normal" if delta <= 0 else "inverse"
        st.metric(
            "Negative Feedback",
            f"{kpis.get('negative_pct', 0):.1f}%",
            delta=delta_str(delta, True),
            delta_color=delta_color
        )
    
    with col6:
        delta = kpis.get('otp_delta', 0)
        st.metric(
            "OTP Verification Rate",
            f"{kpis.get('otp_rate', 0):.1f}%",
            delta=delta_str(delta, True)
        )

# Page rendering functions
def show_overview_page(start_date, end_date, previous_start, previous_end):
    """Display the overview page with KPIs and trends."""
    st.markdown("## 📊 Dashboard Overview")
    
    # Get and display KPIs with error handling
    try:
        kpis = get_kpis(start_date, end_date, previous_start, previous_end)
        display_kpis(kpis)
    except Exception as e:
        st.error(f"Error loading KPIs: {e}")
    
    st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)
    st.markdown("## 📈 Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Messages per day trend
        try:
            daily_messages = run_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as message_count
                FROM chat_messages
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, {'start_date': start_date, 'end_date': end_date})
            
            if not daily_messages.empty:
                fig_messages = px.line(
                    daily_messages, 
                    x='date', 
                    y='message_count',
                    title="Daily Message Volume",
                    labels={'date': 'Date', 'message_count': 'Messages'},
                    color_discrete_sequence=[BRAND_COLORS['primary']]
                )
                fig_messages.update_layout(showlegend=False)
                st.plotly_chart(fig_messages, use_container_width=True)
            else:
                st.info("📊 No message data available for the selected period")
        except Exception as e:
            st.error(f"Error loading message trends: {e}")
    
    with col2:
        # Sessions per day trend
        try:
            daily_sessions = run_query("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(DISTINCT session_id) as session_count
                FROM chat_sessions
                WHERE created_at BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY DATE(created_at)
                ORDER BY date
            """, {'start_date': start_date, 'end_date': end_date})
            
            if not daily_sessions.empty:
                fig_sessions = px.line(
                    daily_sessions, 
                    x='date', 
                    y='session_count',
                    title="Daily New Sessions",
                    labels={'date': 'Date', 'session_count': 'Sessions'},
                    color_discrete_sequence=[BRAND_COLORS['success']]
                )
                fig_sessions.update_layout(showlegend=False)
                st.plotly_chart(fig_sessions, use_container_width=True)
            else:
                st.info("📊 No session data available for the selected period")
        except Exception as e:
            st.error(f"Error loading session trends: {e}")
    
    st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)
    st.markdown("## 🕐 Activity Heatmap")
    
    try:
        activity_data = run_query("""
            SELECT 
                EXTRACT(hour FROM timestamp) as hour,
                EXTRACT(dow FROM timestamp) as day_of_week,
                COUNT(*) as message_count
            FROM chat_messages
            WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY EXTRACT(hour FROM timestamp), EXTRACT(dow FROM timestamp)
            ORDER BY hour, day_of_week
        """, {'start_date': start_date, 'end_date': end_date})
        
        if not activity_data.empty:
            heatmap_data = activity_data.pivot(index='hour', columns='day_of_week', values='message_count').fillna(0)
            day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            # Ensure all days of the week are present
            for i in range(7):
                if i not in heatmap_data.columns:
                    heatmap_data[i] = 0
            heatmap_data = heatmap_data.sort_index(axis=1)
            
            heatmap_data.columns = [day_labels[int(col)] for col in heatmap_data.columns]
            
            fig_heatmap = px.imshow(
                heatmap_data,
                title="Message Activity by Hour and Day of Week",
                labels=dict(x="Day of Week", y="Hour of Day", color="Messages"),
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("📊 No activity data available for heatmap")
    except Exception as e:
        st.error(f"Error loading activity heatmap: {e}")

def show_feedback_page(start_date, end_date, min_rating, max_rating, feedback_type, email_domain):
    """Display the feedback analysis page."""
    st.markdown("## 💬 Feedback Analysis")
    
    # Build filter conditions
    filter_conditions = ["cf.created_at BETWEEN %(start_date)s AND %(end_date)s"]
    filter_params = {'start_date': start_date, 'end_date': end_date}
    
    if min_rating != 1 or max_rating != 5:
        filter_conditions.append("cf.rating::int BETWEEN %(min_rating)s AND %(max_rating)s")
        filter_params.update({'min_rating': min_rating, 'max_rating': max_rating})
    
    if feedback_type != "All":
        filter_conditions.append("cf.feedback_type = %(feedback_type)s")
        filter_params['feedback_type'] = feedback_type
    
    if email_domain:
        filter_conditions.append("cf.email LIKE %(email_pattern)s")
        filter_params['email_pattern'] = f"%@{email_domain}" # More specific search
    
    filter_clause = " AND ".join(filter_conditions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        try:
            rating_dist = run_query(f"""
                SELECT 
                    rating::int as rating,
                    COUNT(*) as count
                FROM chat_feedback cf
                WHERE {filter_clause}
                GROUP BY rating::int
                ORDER BY rating::int
            """, filter_params)
            
            if not rating_dist.empty:
                fig_rating = px.bar(
                    rating_dist,
                    x='rating',
                    y='count',
                    title="Rating Distribution",
                    color='rating',
                    color_continuous_scale=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_rating, use_container_width=True)
            else:
                st.info("📊 No rating data available for the selected criteria")
        except Exception as e:
            st.error(f"Error loading rating distribution: {e}")
    
    with col2:
        # Feedback type distribution
        try:
            type_dist = run_query(f"""
                SELECT 
                    feedback_type,
                    COUNT(*) as count
                FROM chat_feedback cf
                WHERE {filter_clause}
                GROUP BY feedback_type
                ORDER BY count DESC
            """, filter_params)
            
            if not type_dist.empty:
                fig_type = px.pie(
                    type_dist,
                    values='count',
                    names='feedback_type',
                    title="Feedback Type Distribution"
                )
                st.plotly_chart(fig_type, use_container_width=True)
            else:
                st.info("📊 No feedback type data available for the selected criteria")
        except Exception as e:
            st.error(f"Error loading feedback type distribution: {e}")
    
    st.markdown("### 📝 Recent Feedback")
    
    try:
        feedback_data = run_query(f"""
            SELECT 
                cf.created_at,
                cf.session_id,
                cf.email,
                cf.rating::int as rating,
                cf.feedback_type,
                cf.comment
            FROM chat_feedback cf
            WHERE {filter_clause}
            ORDER BY cf.created_at DESC
            LIMIT 100
        """, filter_params)
        
        if not feedback_data.empty:
            st.dataframe(feedback_data, use_container_width=True)
        else:
            st.info("No feedback entries match the current filters.")
    except Exception as e:
        st.error(f"Error loading feedback table: {e}")

# FIX: Added placeholder functions for pages that were not defined.
def show_conversations_page(start_date, end_date):
    st.warning("🚧 The 'Conversations' page is currently under construction.")

def show_sessions_page(start_date, end_date, only_active):
    st.warning("🚧 The 'Sessions' page is currently under construction.")

def show_otp_page(start_date, end_date):
    st.warning("🚧 The 'OTP Monitor' page is currently under construction.")

def show_reports_page(start_date, end_date):
    st.warning("🚧 The 'Reports' page is currently under construction.")


# FIX: Wrapped the main application logic in a function for better structure.
def main():
    """Main application function."""
    load_css()
    
    # Check authentication
    if not check_password():
        return
    
    # Header with logo and branding
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
    
    # Sidebar with logo and filters
    with st.sidebar:
        display_miva_logo(180)
        st.markdown("---")
        st.markdown("## 📊 Dashboard Filters")

    # Period selection
    period_options = ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Custom"]
    period = st.sidebar.selectbox("📅 Time Period", period_options)
    
    # Custom date range for custom period
    custom_start, custom_end = None, None
    if period == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            custom_start = st.date_input("Start Date", value=date.today() - timedelta(days=30))
        with col2:
            custom_end = st.date_input("End Date", value=date.today())
    
    # Compute dates
    start_date, end_date = compute_dates(period, custom_start, custom_end)
    period_length = (end_date - start_date).days
    previous_start = start_date - timedelta(days=period_length)
    previous_end = start_date
    
    # Additional filters
    st.sidebar.markdown("### 🔍 Advanced Filters")
    widget_filter = st.sidebar.text_input("🎯 Widget ID", placeholder="Enter widget ID...")
    rating_range = st.sidebar.slider("⭐ Rating Range", 1, 5, (1, 5))
    min_rating, max_rating = rating_range
    feedback_type_options = ["All", "positive", "negative", "neutral", "bug_report", "suggestion"]
    feedback_type = st.sidebar.selectbox("📝 Feedback Type", feedback_type_options)
    email_domain = st.sidebar.text_input("📧 Email Domain", placeholder="e.g., gmail.com")
    only_active = st.sidebar.checkbox("🟢 Only Active Sessions", value=True)
    
    # Page selection
    st.sidebar.markdown("## 📱 Navigation")
    pages = ["📊 Overview", "💬 Feedback", "🗨️ Conversations", "👥 Sessions", "🔐 OTP Monitor", "📄 Reports"]
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # FIX: Placed page rendering logic in a try/except block for safety.
    try:
        if selected_page == "📊 Overview":
            # FIX: Added the missing call to show the overview page.
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
        st.error(f"An error occurred: {e}")
        st.info("Please try refreshing the page or contact support if the problem persists.")
        
        # Show error details for debugging (only for Owner role)
        if st.session_state.get('user_role') == 'Owner':
            with st.expander("🔧 Debug Information (Owner Only)"):
                st.code(str(e))
                st.write("**Session State:**")
                st.write(st.session_state)

if __name__ == "__main__":
    main()
