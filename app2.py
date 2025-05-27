import streamlit as st

# Import all your modules
from modules.overview_module import show_overview_dashboard
from modules.sentiment_module import show_sentiment_dashboard
# Assuming helpfulness_module.py exists and is correct. If not, you might need to create it.
from modules.helpfulness_module import show_helpfulness_dashboard
from modules.product_user_module import show_product_user_dashboard
from modules.advanced_module import show_advanced_dashboard
from modules.revenue_growth_module import show_revenue_growth_dashboard # <--- NEW MODULE IMPORT

st.set_page_config(
    page_title="Amazon Fine Food Reviews Analytics",
    page_icon="✨",
    layout="wide"
)

# --- Custom CSS for Aesthetic Enhancements ---
st.markdown("""
<style>
/* Import a cool Google Font - Montserrat */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

body {
    font-family: 'Montserrat', sans-serif;
    color: #34495e; /* Dark blue-grey for main text */
}

.stApp {
    background-color: #f8f9fa; /* Very light grey/off-white background */
}

/* Page Title Styling */
h1 {
    font-family: 'Montserrat', sans-serif;
    color: #2c3e50; /* Even darker blue-grey for main titles */
    text-align: center;
    font-weight: 700; /* Bolder title */
    margin-bottom: 30px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.08); /* Slightly stronger text shadow */
    letter-spacing: 1px; /* A bit more spacing for title */
}

/* Introductory Pitch Styling */
div[data-testid="stMarkdownContainer"] p {
    font-family: 'Montserrat', sans-serif;
    color: #5d6d7e;
    line-height: 1.6;
}
div[data-testid="stMarkdownContainer"] p strong {
    color: #2c3e50;
}

/* Style for the overall card-like block (a div that holds content and button) */
.stCardContainer {
    border: 1px solid #e0e0e0; /* Lighter border */
    border-radius: 15px;
    padding: 15px; /* Smaller padding */
    margin-bottom: 20px; /* Consistent spacing */
    background: linear-gradient(135deg, #ffffff, #fefefe); /* Cleaner white gradient */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); /* Softer, spread-out shadow */
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    transition: all 0.2s ease-out; /* Faster, snappier transition */
    height: 180px; /* Fixed height for smaller cards */
    justify-content: space-between; /* Pushes button to bottom */
    font-family: 'Montserrat', sans-serif;
}

/* Hover effect for the entire card container (visual only, for aesthetics) */
.stCardContainer:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); /* More prominent shadow on hover */
    transform: translateY(-7px); /* More pronounced lift effect */
    border: 1px solid #74b9ff; /* Subtle blue border on hover */
}

/* Style for the card title */
.stCardContainer .card-title {
    font-size: 18px; /* Slightly smaller title */
    font-weight: 600; /* Semi-bold */
    color: #34495e; /* Darker title color */
    margin-bottom: 5px; /* Reduced margin */
    line-height: 1.2;
}

/* Style for the card description */
.stCardContainer .card-description {
    font-size: 12px; /* Smaller description text */
    color: #7f8c8d; /* Muted, softer description color */
    line-height: 1.3;
    flex-grow: 1; /* Allows description to take up available space for equal height */
    display: flex; /* For vertical centering of description */
    align-items: center;
    justify-content: center;
    margin-bottom: 10px; /* Space before the button */
}
/* Style for Plotly chart containers */
div[data-testid="stPlotlyChart"] {
    border-radius: 15px; /* Match card border radius */
    box-shadow: 0 4px 15px rgba(0,0,0,0.08); /* Match card shadow */
    transition: all 0.3s ease-in-out; /* Smooth transition */
    margin-bottom: 25px; /* Consistent spacing below charts */
    overflow: hidden; /* Ensures shadow/transform doesn't clip */
    background-color: #ffffff; /* Ensure chart background is white */
    padding: 10px; /* Small padding inside the chart container */
}

div[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.15); /* More prominent shadow on hover */
    transform: translateY(-7px) scale(1.02); /* Lift and subtle zoom for charts */
}

/* Ensure Plotly tooltips are readable */
.plotly-notifier {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    color: #2c3e50 !important;
}

/* Style for the emoji icon */
.stCardContainer .card-icon {
    font-size: 30px; /* Smaller, but still impactful icons */
    margin-bottom: 10px; /* Reduced margin */
    color: #00b894; /* A fresh, vibrant teal/green for icons */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}

/* Style for the actual Streamlit button *inside* the card */
.stCardContainer .stButton > button {
    width: 85%; /* Button wider within the card */
    padding: 7px 10px; /* Smaller padding */
    font-size: 13px; /* Smaller font size for button text */
    font-weight: 600; /* Semi-bold */
    color: white;
    background-color: #4CAF50; /* A classic, slightly brighter green for action */
    border: none;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); /* Subtle shadow for the button itself */
    transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer; /* Explicitly show pointer on button */
}

.stCardContainer .stButton > button:hover {
    background-color: #409C40; /* Darker green on hover */
    transform: translateY(-1px) scale(1.02); /* Slight lift and zoom for the button */
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}

/* Styling for the Back button */
.back-button > button {
    background-color: #6c757d; /* Grey for back button */
    color: white;
    font-size: 16px;
    height: auto;
    padding: 10px 20px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: background-color 0.3s ease, transform 0.3s ease;
    font-weight: 600;
}
.back-button > button:hover {
    background-color: #5a6268;
    transform: translateY(-2px); /* Slight lift for back button */
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

/* Info box styling */
div[data-testid="stInfo"] {
    background-color: #e8f5e9; /* Light green background */
    border-left: 5px solid #4CAF50; /* Green border */
    padding: 15px;
    border-radius: 8px;
    color: #333;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'

# --- Function to navigate back to home ---
def go_home():
    st.session_state.current_view = 'home'

# --- Main Dashboard Logic ---
if st.session_state.current_view == 'home':
    st.title("🌟 Actionable Insights: Amazon Fine Food Reviews")

    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 18px;">
            **Unlock the voice of your customers!**
            <br>
            This interactive dashboard transforms raw review data into **strategic insights** for product, marketing, and customer experience teams.
        </p>
        <p style="font-weight: bold; color: #28a745; font-size: 16px;">
            Your next big decision, powered by data. 👇
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Card Grid ---
    # Function to create a card with content and an internal button
    def create_interactive_card(column_obj, key_suffix, icon, title, description, view_name):
        with column_obj:
            # The entire card structure is an HTML div rendered by st.markdown
            st.markdown(f"""
            <div class="stCardContainer">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-description">{description}</div>
            """, unsafe_allow_html=True) # HTML for the visual part of the card

            # Place the standard Streamlit button *without* unsafe_allow_html.
            if st.button(f"View {title}", key=f'btn_{key_suffix}'):
                st.session_state.current_view = view_name

            # Close the HTML div.
            st.markdown("</div>", unsafe_allow_html=True)


    # Layout with 3 columns for the first row to fit more cards
    col1, col2, col3 = st.columns(3)

    # Render cards using the function
    create_interactive_card(col1, 'overview', '📊', 'Data Overview', 'Key metrics, review distributions, and historical trends for quick insights.', 'overview')
    create_interactive_card(col2, 'sentiment', '😊', 'Sentiment Insights', 'Uncover common themes and emotional drivers in customer feedback for product refinement.', 'sentiment')
    create_interactive_card(col3, 'helpfulness', '👍', 'Helpfulness Deep Dive', 'Identify reviews that truly resonate with customers and understand their characteristics.', 'helpfulness')

    # New row for the remaining cards, ensuring all modules are displayed
    # Now using 3 columns for the second row as well
    col4, col5, col6 = st.columns(3)

    create_interactive_card(col4, 'product_user', '🛍️', 'Product & User Insights', 'Discover top-performing products, highly active users, and influential reviewer patterns.', 'product_user')
    create_interactive_card(col5, 'advanced', '🔬', 'Advanced Analytics', 'Perform custom keyword searches, analyze text length correlations, and explore future analytical directions.', 'advanced')
    create_interactive_card(col6, 'revenue_growth', '📈', 'Revenue Growth', 'Identify high-potential products, analyze performance trends, and uncover strategies for sales growth.', 'revenue_growth') # <--- NEW CARD


    st.markdown("---")
    st.info("💡 **Performance Note:** This dashboard uses intelligent data sampling and caching for lightning-fast responsiveness. For large-scale, real-time analytics, cloud-based solutions are ready to power your enterprise needs.")


# --- Detailed Views (Conditional Rendering) ---
else:
    st.markdown('<div class="back-button">', unsafe_allow_html=True)
    if st.button("⬅️ Back to Dashboard Home", key="back_btn"):
        go_home()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.current_view == 'overview':
        show_overview_dashboard()
    elif st.session_state.current_view == 'sentiment':
        show_sentiment_dashboard()
    elif st.session_state.current_view == 'helpfulness':
        show_helpfulness_dashboard()
    elif st.session_state.current_view == 'product_user':
        show_product_user_dashboard()
    elif st.session_state.current_view == 'advanced':
        show_advanced_dashboard()
    elif st.session_state.current_view == 'revenue_growth': # <--- NEW MODULE LOGIC
        show_revenue_growth_dashboard() # <--- NEW MODULE LOGIC