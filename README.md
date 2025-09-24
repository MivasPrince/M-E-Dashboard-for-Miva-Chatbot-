# Miva Open University - Monitoring & Evaluation Dashboard

A comprehensive Streamlit dashboard for monitoring chatbot conversations, user feedback, session analytics, and OTP verification systems at Miva Open University.

## Features

### 🎯 Core Functionality
- **Authentication**: Secure login system (username: `miva_admin`, password: `password123`)
- **Multi-page Dashboard**: Overview, Feedback, Conversations, Sessions, OTP Monitor, Reports
- **Professional UI**: Brand-consistent design with Miva color palette
- **Real-time Analytics**: Live KPIs with period-over-period comparisons
- **Advanced Filtering**: Date ranges, rating filters, email domains, widget IDs
- **Export Capabilities**: CSV, JSON, and PDF report generation
- **Role-based Access**: Viewer, Analyst, and Owner permission levels

### 📊 Dashboard Pages

#### 1. Overview
- **KPI Tiles**: Active Sessions, Total Messages, Unique Users, CSAT Score, Negative Feedback %, OTP Verification Rate
- **Trend Analysis**: Daily message volume and session creation trends
- **Activity Heatmap**: Message activity by hour and day of week
- **Period Comparisons**: Delta indicators showing changes from previous period

#### 2. Feedback Analysis
- **Rating Distribution**: Visual breakdown of 1-5 star ratings
- **Feedback Type Analysis**: Pie chart of feedback categories (positive, negative, bug reports, etc.)
- **Detailed Feedback Table**: Searchable table with user comments, ratings, and metadata
- **Session Drill-down**: Click-through to view full conversation threads

#### 3. Conversation Explorer
- **Global Search**: Find conversations by Session ID, User ID, or message content
- **Thread Viewer**: Chat bubble interface showing full conversation history
- **Export Options**: Download conversations as CSV or JSON
- **Session Metadata**: Display session context and technical details

#### 4. Session Analytics
- **New vs Returning Users**: Track user retention and engagement patterns
- **Session Duration Analysis**: Histogram of conversation lengths
- **Cohort Analysis**: Segment users by email domain, widget ID, or activity level
- **Widget Performance**: Compare engagement across different chat widgets

#### 5. OTP Monitor
- **Verification Funnel**: Track OTP creation → verification conversion rates
- **Time Analysis**: Distribution of verification times with median metrics
- **Failure Analysis**: Detailed view of failed OTP attempts with retry patterns
- **Security Metrics**: Monitor for suspicious verification patterns

#### 6. Reports & Export
- **Report Generation**: Executive summaries, detailed analytics, and custom reports
- **Multi-format Export**: PDF, CSV, and JSON download options
- **Report Notes**: Add context and annotations to generated reports
- **Access Control**: Export permissions based on user role

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database access
- Required Python packages (see requirements.txt)

### Quick Start

1. **Clone or download the application files**
   ```bash
   # Create project directory
   mkdir miva-dashboard
   cd miva-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure database connection**
   The dashboard is pre-configured to connect to:
   ```
   Host: 16.170.143.253
   Port: 5432
   Database: miva_ai_db
   User: admin
   Password: password123
   ```

4. **Add Miva logo (optional)**
   ```bash
   mkdir assets
   # Place miva-logo.png in the assets folder
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Login with username: `miva_admin`, password: `password123`

## Database Schema

The dashboard works with the following PostgreSQL tables:

### Core Tables
- **chat_messages**: Individual messages in conversations
- **chat_sessions**: Session management and metadata
- **conversation_sessions**: Extended session data with JSONB storage
- **conversation_history**: Aggregated conversation statistics

### Feedback Tables
- **chat_feedback**: User ratings and comments with device metadata
- **user_feedback**: Extended feedback with additional rating fields

### Authentication Tables
- **otps**: One-time password generation
- **otp_verifications**: OTP verification attempts and status

## Configuration

### Brand Customization
The dashboard uses Miva's official brand colors:
- **Primary Blue**: #2563EB (buttons, headers, highlights)
- **Secondary Gray**: #6B7280 (labels, neutral text)
- **White**: #FFFFFF (backgrounds, cards)
- **Accent Red**: #EF4444 (errors, negative metrics)

### User Roles
- **Viewer**: Read-only access to all dashboards
- **Analyst**: Read access + export capabilities + saved filters
- **Owner**: Full access + settings management + user role assignment

### Performance Settings
- **Query Caching**: 5-minute TTL for database queries
- **Data Limits**: Tables limited to 100-500 rows for performance
- **Refresh Rate**: Manual refresh or page navigation

## Usage Guidelines

### Daily Monitoring
1. **Morning Check**: Review Overview page for overnight activity and any alerts
2. **Feedback Review**: Check new ratings and comments in Feedback section
3. **Issue Investigation**: Use Conversation Explorer for specific problem reports

### Weekly Analysis
1. **Trend Analysis**: Compare current week vs previous week metrics
2. **Cohort Performance**: Review email domain and widget performance
3. **Report Generation**: Create executive summary for stakeholder review

### Monthly Deep Dive
1. **Comprehensive Analysis**: Use 30-90 day periods for trend identification
2. **User Journey Analysis**: Track user retention and engagement patterns
3. **System Health**: Review OTP performance and technical metrics

## Troubleshooting

### Common Issues

**Database Connection Failed**
- Verify network connectivity to database server
- Check credentials and database name
- Ensure PostgreSQL service is running

**Slow Query Performance**
- Reduce date range for large datasets
- Use more specific filters to limit result sets
- Check database indexes on timestamp columns

**Authentication Problems**
- Verify username: `miva_admin` and password: `password123`
- Clear browser cache and cookies
- Check for typos in credentials

**Missing Data**
- Verify date ranges include expected activity periods
- Check filter settings that might be excluding data
- Confirm database contains data for selected time range

### Performance Optimization

**For Large Datasets:**
- Use shorter time periods (7-30 days) for initial analysis
- Apply filters before expanding date ranges
- Export large datasets rather than viewing in browser

**For Slow Loading:**
- Clear Streamlit cache using browser refresh (Ctrl+F5)
- Restart the Streamlit application
- Check database server performance

## Security Considerations

### Data Protection
- IP addresses are masked by default in exports
- No PII (personally identifiable information) included in downloads
- Session IDs are anonymized in reports

### Access Control
- Authentication required for all dashboard access
- Role-based permissions for export functionality
- Audit logging for report generation (planned feature)

### Database Security
- Read-only database access recommended for production
- Use environment variables for sensitive configuration
- Regular password rotation for database credentials

## Development & Customization

### Adding New Metrics
1. Create SQL query in the appropriate page function
2. Add visualization using Plotly
3. Update KPI calculation if needed
4. Test with sample data

### Custom Branding
1. Replace logo in `/assets/miva-logo.png`
2. Update `BRAND_COLORS` dictionary in main application
3. Modify CSS styling in `load_css()` function

### New Dashboard Pages
1. Create new page function following existing patterns
2. Add page to navigation menu
3. Update routing in `main()` function
4. Test all functionality

## Support & Maintenance

### Regular Maintenance
- **Weekly**: Review dashboard performance and user feedback
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Database cleanup and optimization

### Monitoring
- Track dashboard usage and performance metrics
- Monitor database query performance
- Review user access patterns and requirements

### Updates
- Follow semantic versioning for releases
- Test all changes in staging environment
- Document breaking changes and migration steps

## Technical Specifications

### System Requirements
- **Python**: 3.8+ (recommended 3.10+)
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB for application and dependencies
- **Network**: Stable connection to PostgreSQL database

### Dependencies
- **Streamlit**: Web application framework
- **psycopg2**: PostgreSQL database adapter
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations

### Browser Support
- **Chrome**: 90+ (recommended)
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## License & Attribution

This dashboard is developed for Miva Open University's internal use. The application uses open-source libraries and follows best practices for data visualization and user experience.

For questions, support, or feature requests, please contact the Student Success team or IT Department.
