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
from typing import Dict, Any, Optional, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Miva M&E Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration
DB_CONFIG = {
    "host": "16.170.143.253",
    "port": 5432,
    "user": "admin",
    "password": "password123",
    "database": "miva_ai_db",
}

# Brand colors
BRAND_COLORS = {
    "primary": "#2563EB",
    "secondary": "#6B7280", 
    "white": "#FFFFFF",
    "accent": "#EF4444",
    "success": "#10B981",
    "warning": "#F59E0B"
}

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
        
        if username == "miva_admin" and password == "password123":
            st.session_state["password_correct"] = True
            st.session_state["user_role"] = "Owner"  # Set role for authenticated user
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 60vh;">
        <div style="background: white; padding: 3rem; border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1); max-width: 400px; width: 100%;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="color: {BRAND_COLORS['primary']}; margin-bottom: 0.5rem;">Miva Open University</h1>
                <h3 style="color: {BRAND_COLORS['secondary']}; margin-bottom: 2rem;">M&E Dashboard</h3>
            </div>
    """, unsafe_allow_html=True)
    
    st.text_input("Username", key="username", placeholder="Enter username")
    st.text_input("Password", type="password", key="password", placeholder="Enter password")
    st.button("Login", on_click=password_entered, use_container_width=True)
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😞 Username or password incorrect")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    return False

# Database connection functions
@st.cache_resource
def get_connection():
    """Get database connection with caching."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_query(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

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
    if delta is None or delta == 0:
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
            'sessions_delta': current['active_sessions'] - previous['active_sessions'],
            'messages_delta': current['total_messages'] - previous['total_messages'],
            'users_delta': current['unique_users'] - previous['unique_users'],
            'csat_delta': current['csat'] - previous['csat'],
            'negative_delta': current['negative_pct'] - previous['negative_pct'],
            'otp_delta': current['otp_rate'] - previous['otp_rate']
        }
    else:
        deltas = {
            'sessions_delta': 0, 'messages_delta': 0, 'users_delta': 0,
            'csat_delta': 0, 'negative_delta': 0, 'otp_delta': 0
        }
    
    return {**current.to_dict(), **deltas}

def display_kpis(kpis: Dict[str, Any]):
    """Display KPI metrics in a professional layout."""
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        delta = kpis.get('sessions_delta', 0)
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        st.metric(
            "Active Sessions",
            f"{int(kpis.get('active_sessions', 0)):,}",
            delta=delta_str(delta),
            delta_color=delta_color
        )
    
    with col2:
        delta = kpis.get('messages_delta', 0)
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        st.metric(
            "Total Messages",
            f"{int(kpis.get('total_messages', 0)):,}",
            delta=delta_str(delta),
            delta_color=delta_color
        )
    
    with col3:
        delta = kpis.get('users_delta', 0)
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        st.metric(
            "Unique Users",
            f"{int(kpis.get('unique_users', 0)):,}",
            delta=delta_str(delta),
            delta_color=delta_color
        )
    
    with col4:
        delta = kpis.get('csat_delta', 0)
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        st.metric(
            "CSAT Score",
            f"{kpis.get('csat', 0):.2f}/5.0",
            delta=delta_str(delta),
            delta_color=delta_color
        )
    
    with col5:
        delta = kpis.get('negative_delta', 0)
        delta_color = "normal" if delta == 0 else ("normal" if delta < 0 else "inverse")  # Negative is good for negative feedback
        st.metric(
            "Negative Feedback",
            f"{kpis.get('negative_pct', 0):.1f}%",
            delta=delta_str(delta, True),
            delta_color=delta_color
        )
    
    with col6:
        delta = kpis.get('otp_delta', 0)
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        st.metric(
            "OTP Verification Rate",
            f"{kpis.get('otp_rate', 0):.1f}%",
            delta=delta_str(delta, True),
            delta_color=delta_color
        )

def main():
    """Main application function."""
    load_css()
    
    # Check authentication
    if not check_password():
        return
    
    # Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #E5E7EB;">
        <div>
            <h1 class="header-title">Miva Open University</h1>
            <p class="subheader">Monitoring & Evaluation Dashboard</p>
        </div>
        <div style="margin-left: auto; background: {BRAND_COLORS['primary']}; color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; font-size: 0.875rem;">
            Role: {st.session_state.get('user_role', 'Viewer')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("## 📊 Dashboard Filters")
    
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
    
    # Display selected page
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

def show_overview_page(start_date, end_date, previous_start, previous_end):
    """Display the overview page with KPIs and trends."""
    st.markdown("## 📊 Dashboard Overview")
    
    # Get and display KPIs
    kpis = get_kpis(start_date, end_date, previous_start, previous_end)
    display_kpis(kpis)
    
    # Trends section
    st.markdown("## 📈 Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Messages per day trend
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
                color_discrete_sequence=[BRAND_COLORS['primary']]
            )
            fig_messages.update_layout(
                showlegend=False,
                title_font_color=BRAND_COLORS['primary'],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_messages, use_container_width=True)
        else:
            st.info("No message data available for the selected period")
    
    with col2:
        # Sessions per day trend
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
                color_discrete_sequence=[BRAND_COLORS['success']]
            )
            fig_sessions.update_layout(
                showlegend=False,
                title_font_color=BRAND_COLORS['primary'],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sessions, use_container_width=True)
        else:
            st.info("No session data available for the selected period")
    
    # Activity heatmap
    st.markdown("## 🕐 Activity Heatmap")
    
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
        # Create pivot table for heatmap
        heatmap_data = activity_data.pivot(index='hour', columns='day_of_week', values='message_count').fillna(0)
        
        # Day labels
        day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        heatmap_data.columns = [day_labels[int(col)] for col in heatmap_data.columns]
        
        fig_heatmap = px.imshow(
            heatmap_data,
            title="Message Activity by Hour and Day of Week",
            labels=dict(x="Day of Week", y="Hour of Day", color="Messages"),
            color_continuous_scale="Blues"
        )
        fig_heatmap.update_layout(
            title_font_color=BRAND_COLORS['primary']
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No activity data available for heatmap")

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
        filter_params['email_pattern'] = f"%{email_domain}"
    
    filter_clause = " AND ".join(filter_conditions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
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
                color='count',
                color_continuous_scale="RdYlBu_r"
            )
            fig_rating.update_layout(
                title_font_color=BRAND_COLORS['primary'],
                showlegend=False
            )
            st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Feedback type distribution
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
            fig_type.update_layout(
                title_font_color=BRAND_COLORS['primary']
            )
            st.plotly_chart(fig_type, use_container_width=True)
    
    # Feedback table with details
    st.markdown("### 📝 Recent Feedback")
    
    feedback_data = run_query(f"""
        SELECT 
            cf.created_at,
            cf.session_id,
            cf.email,
            cf.rating::int as rating,
            cf.feedback_type,
            cf.comment,
            cf.user_agent,
            cf.ip_address
        FROM chat_feedback cf
        WHERE {filter_clause}
        ORDER BY cf.created_at DESC
        LIMIT 100
    """, filter_params)
    
    if not feedback_data.empty:
        # Format the data for display
        display_df = feedback_data.copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['rating_stars'] = display_df['rating'].apply(lambda x: '⭐' * x + '☆' * (5-x))
        display_df['comment_preview'] = display_df['comment'].apply(
            lambda x: (str(x)[:50] + '...') if pd.notna(x) and len(str(x)) > 50 else str(x) if pd.notna(x) else ''
        )
        
        # Select columns for display
        columns_to_show = ['created_at', 'email', 'rating_stars', 'feedback_type', 'comment_preview', 'session_id']
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
            column_config={
                'created_at': 'Date/Time',
                'email': 'User Email',
                'rating_stars': 'Rating',
                'feedback_type': 'Type',
                'comment_preview': 'Comment',
                'session_id': st.column_config.LinkColumn(
                    'Session',
                    help="Click to view full conversation",
                    display_text="View"
                )
            }
        )
        
        # Session detail drawer
        if st.button("🔍 View Session Details", help="Select a session from the table above"):
            session_id = st.text_input("Enter Session ID:", placeholder="Paste session ID here")
            if session_id:
                show_session_conversation(session_id)
    else:
        st.info("No feedback data available for the selected criteria")

def show_session_conversation(session_id: str):
    """Display full conversation for a session."""
    st.markdown(f"### 🗨️ Conversation Details - Session: `{session_id}`")
    
    # Get conversation messages
    messages = run_query("""
        SELECT 
            message_type,
            content,
            timestamp,
            message_metadata
        FROM chat_messages
        WHERE session_id = %(session_id)s
        ORDER BY timestamp ASC
    """, {'session_id': session_id})
    
    if not messages.empty:
        # Display messages in chat format
        for _, msg in messages.iterrows():
            timestamp = pd.to_datetime(msg['timestamp']).strftime('%H:%M:%S')
            msg_type = msg['message_type'].lower()
            content = msg['content']
            
            if msg_type == 'user':
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                    <div class="conversation-bubble user-bubble">
                        <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 5px;">User • {timestamp}</div>
                        <div>{content}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif msg_type == 'bot' or msg_type == 'assistant':
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div class="conversation-bubble bot-bubble">
                        <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 5px;">Assistant • {timestamp}</div>
                        <div>{content}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: center; margin: 10px 0;">
                    <div class="conversation-bubble system-bubble">
                        <div style="font-size: 0.8em; margin-bottom: 5px;">System • {timestamp}</div>
                        <div>{content}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"No messages found for session ID: {session_id}")

def show_conversations_page(start_date, end_date):
    """Display the conversations exploration page."""
    st.markdown("## 🗨️ Conversation Explorer")
    
    # Search functionality
    st.markdown("### 🔍 Search Conversations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_session_id = st.text_input("🆔 Session ID", placeholder="Enter session ID...")
    
    with col2:
        search_user_id = st.text_input("👤 User ID", placeholder="Enter user ID...")
    
    with col3:
        search_content = st.text_input("💬 Message Content", placeholder="Search in messages...")
    
    # Build search query
    search_conditions = ["cm.timestamp BETWEEN %(start_date)s AND %(end_date)s"]
    search_params = {'start_date': start_date, 'end_date': end_date}
    
    if search_session_id:
        search_conditions.append("cm.session_id = %(session_id)s")
        search_params['session_id'] = search_session_id
    
    if search_user_id:
        search_conditions.append("(cs.user_id = %(user_id)s OR ccs.user_id = %(user_id)s)")
        search_params['user_id'] = search_user_id
    
    if search_content:
        search_conditions.append("cm.content ILIKE %(content_pattern)s")
        search_params['content_pattern'] = f"%{search_content}%"
    
    search_clause = " AND ".join(search_conditions)
    
    # Get conversation sessions
    conversations = run_query(f"""
        SELECT DISTINCT
            cm.session_id,
            COALESCE(cs.user_id, ccs.user_id) as user_id,
            MIN(cm.timestamp) as first_message,
            MAX(cm.timestamp) as last_message,
            COUNT(cm.id) as message_count,
            STRING_AGG(
                CASE WHEN cm.message_type = 'user' 
                THEN LEFT(cm.content, 50) 
                END, ' | ' ORDER BY cm.timestamp
            ) as user_messages_preview
        FROM chat_messages cm
        LEFT JOIN chat_sessions cs ON cm.session_id = cs.session_id
        LEFT JOIN conversation_sessions ccs ON cm.session_id = ccs.session_id
        WHERE {search_clause}
        GROUP BY cm.session_id, COALESCE(cs.user_id, ccs.user_id)
        ORDER BY MAX(cm.timestamp) DESC
        LIMIT 50
    """, search_params)
    
    if not conversations.empty:
        st.markdown("### 📋 Search Results")
        
        # Format display data
        display_conversations = conversations.copy()
        display_conversations['first_message'] = pd.to_datetime(display_conversations['first_message']).dt.strftime('%Y-%m-%d %H:%M')
        display_conversations['last_message'] = pd.to_datetime(display_conversations['last_message']).dt.strftime('%Y-%m-%d %H:%M')
        display_conversations['duration'] = (
            pd.to_datetime(conversations['last_message']) - pd.to_datetime(conversations['first_message'])
        ).dt.total_seconds() / 60  # Duration in minutes
        display_conversations['duration_str'] = display_conversations['duration'].apply(
            lambda x: f"{int(x//60)}h {int(x%60)}m" if x >= 60 else f"{int(x)}m"
        )
        
        # Display results
        st.dataframe(
            display_conversations[['session_id', 'user_id', 'first_message', 'message_count', 'duration_str', 'user_messages_preview']],
            use_container_width=True,
            column_config={
                'session_id': 'Session ID',
                'user_id': 'User ID',
                'first_message': 'Started',
                'message_count': 'Messages',
                'duration_str': 'Duration',
                'user_messages_preview': 'Preview'
            }
        )
        
        # Session viewer
        selected_session = st.selectbox(
            "🔍 Select session to view full conversation:",
            options=[''] + conversations['session_id'].tolist(),
            format_func=lambda x: f"Session: {x}" if x else "Select a session..."
        )
        
        if selected_session:
            show_session_conversation(selected_session)
            
            # Export options
            if st.button("📥 Export Conversation"):
                export_conversation(selected_session)
    else:
        st.info("No conversations found matching your search criteria")

def export_conversation(session_id: str):
    """Export conversation to CSV/JSON."""
    messages = run_query("""
        SELECT 
            session_id,
            message_type,
            content,
            timestamp,
            message_metadata
        FROM chat_messages
        WHERE session_id = %(session_id)s
        ORDER BY timestamp ASC
    """, {'session_id': session_id})
    
    if not messages.empty:
        # CSV export
        csv_buffer = BytesIO()
        messages.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"conversation_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # JSON export
        json_data = messages.to_json(orient='records', date_format='iso', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"conversation_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def show_sessions_page(start_date, end_date, only_active):
    """Display the sessions analytics page."""
    st.markdown("## 👥 Session Analytics")
    
    # Session activity trends
    col1, col2 = st.columns(2)
    
    with col1:
        # New vs returning sessions
        session_data = run_query("""
            WITH user_sessions AS (
                SELECT 
                    COALESCE(cs.user_id, ccs.user_id) as user_id,
                    DATE(COALESCE(cs.created_at, ccs.created_at)) as session_date,
                    COUNT(*) as sessions_per_day,
                    ROW_NUMBER() OVER (PARTITION BY COALESCE(cs.user_id, ccs.user_id) ORDER BY DATE(COALESCE(cs.created_at, ccs.created_at))) as session_rank
                FROM chat_sessions cs
                FULL OUTER JOIN conversation_sessions ccs ON cs.session_id = ccs.session_id
                WHERE COALESCE(cs.created_at, ccs.created_at) BETWEEN %(start_date)s AND %(end_date)s
                    AND (%(only_active)s = false OR COALESCE(ccs.is_active, true) = true)
                GROUP BY COALESCE(cs.user_id, ccs.user_id), DATE(COALESCE(cs.created_at, ccs.created_at))
            )
            SELECT 
                session_date,
                COUNT(CASE WHEN session_rank = 1 THEN 1 END) as new_users,
                COUNT(CASE WHEN session_rank > 1 THEN 1 END) as returning_users
            FROM user_sessions
            WHERE user_id IS NOT NULL
            GROUP BY session_date
            ORDER BY session_date
        """, {
            'start_date': start_date, 
            'end_date': end_date,
            'only_active': only_active
        })
        
        if not session_data.empty:
            fig_sessions = go.Figure()
            
            fig_sessions.add_trace(go.Scatter(
                x=session_data['session_date'],
                y=session_data['new_users'],
                mode='lines+markers',
                name='New Users',
                line=dict(color=BRAND_COLORS['primary'])
            ))
            
            fig_sessions.add_trace(go.Scatter(
                x=session_data['session_date'],
                y=session_data['returning_users'],
                mode='lines+markers',
                name='Returning Users',
                line=dict(color=BRAND_COLORS['success'])
            ))
            
            fig_sessions.update_layout(
                title="New vs Returning Users",
                title_font_color=BRAND_COLORS['primary'],
                xaxis_title="Date",
                yaxis_title="User Count"
            )
            
            st.plotly_chart(fig_sessions, use_container_width=True)
    
    with col2:
        # Session duration distribution
        duration_data = run_query("""
            SELECT 
                EXTRACT(EPOCH FROM (cs.last_activity - cs.created_at))/60 as duration_minutes
            FROM chat_sessions cs
            WHERE cs.created_at BETWEEN %(start_date)s AND %(end_date)s
                AND cs.last_activity > cs.created_at
                AND EXTRACT(EPOCH FROM (cs.last_activity - cs.created_at))/60 < 180  -- Filter extreme outliers
        """, {'start_date': start_date, 'end_date': end_date})
        
        if not duration_data.empty:
            fig_duration = px.histogram(
                duration_data,
                x='duration_minutes',
                nbins=20,
                title="Session Duration Distribution",
                labels={'duration_minutes': 'Duration (minutes)', 'count': 'Sessions'}
            )
            fig_duration.update_layout(
                title_font_color=BRAND_COLORS['primary'],
                showlegend=False
            )
            fig_duration.update_traces(marker_color=BRAND_COLORS['secondary'])
            
            st.plotly_chart(fig_duration, use_container_width=True)
    
    # Cohort analysis
    st.markdown("### 📊 User Cohorts")
    
    cohort_options = ["Email Domain", "Widget ID", "User Activity"]
    selected_cohort = st.selectbox("Select Cohort Analysis:", cohort_options)
    
    if selected_cohort == "Email Domain":
        cohort_data = run_query("""
            SELECT 
                SPLIT_PART(cf.email, '@', 2) as email_domain,
                COUNT(DISTINCT cf.session_id) as sessions,
                COUNT(DISTINCT cf.user_id) as users,
                AVG(cf.rating::numeric) as avg_rating
            FROM chat_feedback cf
            WHERE cf.created_at BETWEEN %(start_date)s AND %(end_date)s
                AND cf.email IS NOT NULL
                AND cf.email != 'unknown@example.com'
            GROUP BY SPLIT_PART(cf.email, '@', 2)
            HAVING COUNT(DISTINCT cf.session_id) >= 5
            ORDER BY sessions DESC
            LIMIT 15
        """, {'start_date': start_date, 'end_date': end_date})
        
        if not cohort_data.empty:
            fig_cohort = px.bar(
                cohort_data,
                x='email_domain',
                y='sessions',
                title="Sessions by Email Domain",
                color='avg_rating',
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=3
            )
            fig_cohort.update_layout(
                title_font_color=BRAND_COLORS['primary'],
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_cohort, use_container_width=True)
    
    elif selected_cohort == "Widget ID":
        widget_data = run_query("""
            SELECT 
                cf.widget_id,
                COUNT(DISTINCT cf.session_id) as sessions,
                COUNT(cf.id) as feedback_count,
                AVG(cf.rating::numeric) as avg_rating,
                COUNT(CASE WHEN cf.rating::int <= 2 THEN 1 END) * 100.0 / COUNT(*) as negative_pct
            FROM chat_feedback cf
            WHERE cf.created_at BETWEEN %(start_date)s AND %(end_date)s
                AND cf.widget_id IS NOT NULL
            GROUP BY cf.widget_id
            ORDER BY sessions DESC
            LIMIT 10
        """, {'start_date': start_date, 'end_date': end_date})
        
        if not widget_data.empty:
            fig_widget = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Sessions by Widget", "Average Rating by Widget"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_widget.add_trace(
                go.Bar(x=widget_data['widget_id'], y=widget_data['sessions'], name="Sessions"),
                row=1, col=1
            )
            
            fig_widget.add_trace(
                go.Bar(x=widget_data['widget_id'], y=widget_data['avg_rating'], name="Avg Rating"),
                row=1, col=2
            )
            
            fig_widget.update_layout(
                title_text="Widget Performance Analysis",
                title_font_color=BRAND_COLORS['primary'],
                showlegend=False
            )
            
            st.plotly_chart(fig_widget, use_container_width=True)

def show_otp_page(start_date, end_date):
    """Display the OTP monitoring page."""
    st.markdown("## 🔐 OTP Verification Monitor")
    
    # OTP funnel metrics
    col1, col2, col3, col4 = st.columns(4)
    
    otp_metrics = run_query("""
        SELECT 
            COUNT(DISTINCT o.id) as otp_created,
            COUNT(DISTINCT CASE WHEN ov.is_verified THEN ov.id END) as otp_verified,
            COUNT(DISTINCT CASE WHEN NOT ov.is_verified THEN ov.id END) as otp_failed,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY EXTRACT(EPOCH FROM (ov.verified_at - ov.created_at))
            ) as median_verify_time
        FROM otps o
        LEFT JOIN otp_verifications ov ON ov.user_identifier = o.user_id 
            AND ov.otp_code = o.otp_code
        WHERE o.created_at BETWEEN %(start_date)s AND %(end_date)s
    """, {'start_date': start_date, 'end_date': end_date})
    
    if not otp_metrics.empty:
        metrics = otp_metrics.iloc[0]
        
        with col1:
            st.metric("OTPs Created", f"{int(metrics.get('otp_created', 0)):,}")
        
        with col2:
            verified = int(metrics.get('otp_verified', 0))
            st.metric("OTPs Verified", f"{verified:,}")
        
        with col3:
            failed = int(metrics.get('otp_failed', 0))
            st.metric("OTPs Failed", f"{failed:,}")
        
        with col4:
            verify_time = metrics.get('median_verify_time', 0)
            if verify_time:
                time_str = f"{int(verify_time//60)}m {int(verify_time%60)}s"
            else:
                time_str = "N/A"
            st.metric("Median Verify Time", time_str)
    
    # OTP funnel visualization
    col1, col2 = st.columns(2)
    
    with col1:
        if not otp_metrics.empty:
            created = int(otp_metrics.iloc[0].get('otp_created', 0))
            verified = int(otp_metrics.iloc[0].get('otp_verified', 0))
            
            if created > 0:
                funnel_data = pd.DataFrame({
                    'Stage': ['Created', 'Verified'],
                    'Count': [created, verified],
                    'Percentage': [100, (verified/created)*100 if created > 0 else 0]
                })
                
                fig_funnel = px.funnel(
                    funnel_data,
                    x='Count',
                    y='Stage',
                    title="OTP Verification Funnel",
                    color='Percentage',
                    color_continuous_scale="RdYlGn"
                )
                fig_funnel.update_layout(title_font_color=BRAND_COLORS['primary'])
                st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # Verification time distribution
        verify_times = run_query("""
            SELECT 
                EXTRACT(EPOCH FROM (verified_at - created_at))/60 as verify_minutes
            FROM otp_verifications
            WHERE is_verified = true 
                AND verified_at IS NOT NULL
                AND created_at BETWEEN %(start_date)s AND %(end_date)s
                AND EXTRACT(EPOCH FROM (verified_at - created_at))/60 <= 30  -- Filter extreme outliers
        """, {'start_date': start_date, 'end_date': end_date})
        
        if not verify_times.empty:
            fig_verify_time = px.box(
                verify_times,
                y='verify_minutes',
                title="Verification Time Distribution",
                labels={'verify_minutes': 'Time to Verify (minutes)'}
            )
            fig_verify_time.update_layout(
                title_font_color=BRAND_COLORS['primary'],
                showlegend=False
            )
            fig_verify_time.update_traces(marker_color=BRAND_COLORS['accent'])
            st.plotly_chart(fig_verify_time, use_container_width=True)
    
    # Failed OTP analysis
    st.markdown("### ❌ Failed OTP Analysis")
    
    failed_otps = run_query("""
        SELECT 
            ov.user_identifier,
            ov.otp_code,
            ov.attempts,
            ov.created_at,
            ov.expires_at,
            ov.otp_metadata
        FROM otp_verifications ov
        WHERE ov.is_verified = false
            AND ov.created_at BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY ov.created_at DESC
        LIMIT 50
    """, {'start_date': start_date, 'end_date': end_date})
    
    if not failed_otps.empty:
        display_failed = failed_otps.copy()
        display_failed['created_at'] = pd.to_datetime(display_failed['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_failed['expires_at'] = pd.to_datetime(display_failed['expires_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_failed[['created_at', 'user_identifier', 'attempts', 'expires_at']],
            use_container_width=True,
            column_config={
                'created_at': 'Created',
                'user_identifier': 'User',
                'attempts': 'Attempts',
                'expires_at': 'Expires'
            }
        )
    else:
        st.info("No failed OTP verifications in the selected period")

def show_reports_page(start_date, end_date):
    """Display the reports and export page."""
    st.markdown("## 📄 Reports & Export")
    
    user_role = st.session_state.get('user_role', 'Viewer')
    
    if user_role == 'Viewer':
        st.warning("🔒 You have read-only access. Contact an Administrator to get export permissions.")
        return
    
    # Report generation options
    st.markdown("### 📊 Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analytics", "Feedback Report", "OTP Analysis", "Custom"]
        )
    
    with col2:
        export_format = st.selectbox("Export Format", ["PDF", "CSV", "JSON"])
    
    # Report notes
    report_notes = st.text_area("Report Notes", placeholder="Add any notes or context for this report...")
    
    # Generate report button
    if st.button("📋 Generate Report", type="primary"):
        generate_report(report_type, export_format, start_date, end_date, report_notes)
    
    # Saved reports section
    st.markdown("### 💾 Saved Reports")
    st.info("Report history feature would be implemented with a reports table in the database")

def generate_report(report_type: str, export_format: str, start_date: datetime, end_date: datetime, notes: str):
    """Generate and download report."""
    
    # Calculate period for previous comparison
    period_length = (end_date - start_date).days
    previous_start = start_date - timedelta(days=period_length)
    previous_end = start_date
    
    # Get KPIs
    kpis = get_kpis(start_date, end_date, previous_start, previous_end)
    
    if export_format == "CSV":
        # Generate CSV report
        report_data = []
        
        if report_type == "Executive Summary":
            report_data = [
                ["Metric", "Value", "Previous Period", "Change"],
                ["Active Sessions", f"{int(kpis.get('active_sessions', 0)):,}", "", f"{kpis.get('sessions_delta', 0)}"],
                ["Total Messages", f"{int(kpis.get('total_messages', 0)):,}", "", f"{kpis.get('messages_delta', 0)}"],
                ["Unique Users", f"{int(kpis.get('unique_users', 0)):,}", "", f"{kpis.get('users_delta', 0)}"],
                ["CSAT Score", f"{kpis.get('csat', 0):.2f}", "", f"{kpis.get('csat_delta', 0):.2f}"],
                ["Negative Feedback %", f"{kpis.get('negative_pct', 0):.1f}%", "", f"{kpis.get('negative_delta', 0):.1f}%"],
                ["OTP Verification Rate", f"{kpis.get('otp_rate', 0):.1f}%", "", f"{kpis.get('otp_delta', 0):.1f}%"]
            ]
        
        # Convert to DataFrame and CSV
        df_report = pd.DataFrame(report_data[1:], columns=report_data[0])
        csv_buffer = BytesIO()
        
        # Add metadata
        metadata = [
            f"# Miva Open University - {report_type}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"# Notes: {notes}",
            ""
        ]
        
        csv_content = "\n".join(metadata) + df_report.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_content,
            file_name=f"miva_report_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    elif export_format == "JSON":
        # Generate JSON report
        report_json = {
            "report_metadata": {
                "type": report_type,
                "generated_at": datetime.now().isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "notes": notes,
                "generated_by": f"miva_admin ({st.session_state.get('user_role', 'Unknown')})"
            },
            "kpis": kpis
        }
        
        json_data = json.dumps(report_json, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON Report",
            data=json_data,
            file_name=f"miva_report_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:  # PDF
        st.info("PDF export would require additional libraries like reportlab or weasyprint. For now, please use CSV or JSON export.")
    
    st.success(f"✅ {report_type} report generated successfully!")

if __name__ == "__main__":
    main()
