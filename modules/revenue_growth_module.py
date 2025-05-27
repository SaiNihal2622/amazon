import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data_revenue():
    try:
        df = pd.read_csv('Reviews.csv')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        # Convert 'Time' to datetime, handling potential non-integer values gracefully
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce') 
        
        # Calculate HelpfulnessRatio: if denominator is 0, ratio is 0
        df['HelpfulnessRatio'] = df.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)

        # Ensure ProductId and Score columns exist and are not empty for core calculations
        if 'ProductId' not in df.columns or df['ProductId'].isnull().all():
            st.error("Error: 'ProductId' column is missing or entirely empty in Reviews.csv. Cannot generate revenue insights.")
            st.stop()
        if 'Score' not in df.columns or df['Score'].isnull().all():
            st.error("Error: 'Score' column is missing or entirely empty in Reviews.csv. Cannot generate revenue insights.")
            st.stop()
        if df.empty:
            st.warning("The Reviews.csv file is empty after loading and processing. No insights can be generated.")
            st.stop()

        return df
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data for Revenue Growth: {e}")
        st.stop()


def show_revenue_growth_dashboard():
    st.title("📈 Revenue Drivers & Growth Opportunities")
    st.markdown("---")

    df = load_data_revenue()

    if df is not None:
        # --- PRE-CALCULATE PRODUCT PERFORMANCE METRICS ---
        product_performance = df.groupby('ProductId').agg(
            AverageScore=('Score', 'mean'),
            ReviewCount=('ProductId', 'count'),
            AverageHelpfulnessRatio=('HelpfulnessRatio', 'mean')
        ).reset_index()

        # Filter out products with very few reviews for meaningful analysis
        min_reviews_threshold = 20 
        product_performance = product_performance[product_performance['ReviewCount'] >= min_reviews_threshold].copy()

        if product_performance.empty:
            st.warning("No products meet the minimum review threshold for analysis. Please ensure your Reviews.csv has sufficient data.")
            return

        # Define quartiles for classification (using 75th and 25th percentiles as thresholds)
        score_high_threshold = product_performance['AverageScore'].quantile(0.75)
        score_low_threshold = product_performance['AverageScore'].quantile(0.25)
        review_count_high_threshold = product_performance['ReviewCount'].quantile(0.75)
        review_count_low_threshold = product_performance['ReviewCount'].quantile(0.25)

        # --- DEBUGGING STEP: PRINT THRESHOLDS ---
        st.sidebar.subheader("Classification Thresholds (< 25 Percentile for Low and > 75 Percentile for High)")
        st.sidebar.write(f"Average Score (High): {score_high_threshold:.2f}")
        st.sidebar.write(f"Average Score (Low): {score_low_threshold:.2f}")
        st.sidebar.write(f"Review Count (High): {review_count_high_threshold:.0f}")
        st.sidebar.write(f"Review Count (Low): {review_count_low_threshold:.0f}")
        st.sidebar.markdown("---")
        # --- END DEBUGGING STEP ---


        # Classify products into quadrants based on these thresholds
        def classify_product(row):
            # IMPORTANT: Re-evaluate these conditions based on the printed thresholds.
            # The definitions below are the logical ones, but the thresholds determine the outcome.
            if row['AverageScore'] >= score_high_threshold and row['ReviewCount'] >= review_count_high_threshold:
                return 'Cash Cow (High Score, High Volume)'
            elif row['AverageScore'] < score_low_threshold and row['ReviewCount'] >= review_count_high_threshold:
                return 'Problem Child (Low Score, High Volume)'
            elif row['AverageScore'] >= score_high_threshold and row['ReviewCount'] < review_count_low_threshold:
                return 'Hidden Gem (High Score, Low Volume)'
            else: # All other cases, including low-low and mid-mid, or any combination not explicitly covered
                return 'Underperformer (Low Score, Low Volume)'
        
        product_performance['Category'] = product_performance.apply(classify_product, axis=1)
        
        # Increased font size using HTML style for the main classification description
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <div data-testid="stInfo" style="background-color: #f0f2f6; border-left: 5px solid #6c757d; padding: 10px; border-radius: 5px; color: #333; font-size: 1.2em;"> 
                Products are classified by <b>average review score</b> (quality) and <b>total review volume</b> (popularity). Each category offers unique strategic actions for revenue optimization.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # Display 20 items, but set dataframe height for 5 visible rows
        items_to_display_in_table = 20 # Show 20 products
        dataframe_height = 220 # Approximate height for 5 rows + header

        # Define a consistent, modern color palette for the categories
        category_colors = {
            'Cash Cow (High Score, High Volume)': '#4CAF50',  # Green
            'Problem Child (Low Score, High Volume)': '#DC3545', # Red
            'Hidden Gem (High Score, Low Volume)': '#007BFF',   # Blue
            'Underperformer (Low Score, Low Volume)': '#6C757D' # Gray/Purple
        }

        # New font size for insights (kept separate for consistency within insight boxes)
        insight_font_size = "0.95em" 

        # --- Table Styling Function ---
        def style_dataframe(df_to_style, header_bg_color, header_text_color):
            return df_to_style.style.set_properties(**{
                'font-size': '0.9em',
                'border-collapse': 'collapse',
                'border': '1px solid #ddd'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', header_bg_color),
                    ('color', header_text_color),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '8px')
                ]},
                {'selector': 'td', 'props': [
                    ('padding', '8px'),
                    ('text-align', 'center'),
                    ('border-bottom', '1px solid #eee')
                ]},
                {'selector': 'tr:nth-of-type(odd)', 'props': [ # Zebra striping
                    ('background-color', '#f9f9f9')
                ]},
                {'selector': 'tr:hover', 'props': [ # Hover effect
                    ('background-color', '#f0f0f0')
                ]}
            ])


        with col_left:
            st.write("#### Products for Immediate Consideration")
            
            # --- Problem Children ---
            problem_children = product_performance[product_performance['Category'] == 'Problem Child (Low Score, High Volume)']
            st.markdown(f"""
            <h3 style="color: {category_colors['Problem Child (Low Score, High Volume)']}; font-size: 2em; text-align: left; margin-bottom: 0px;">
                {len(problem_children)} 'Problem Children'
            </h3>
            <p style="font-size: 1.2em; margin-top: 0px; color: #555;"><b>High Review Volume, Low Average Score</b></p>
            """, unsafe_allow_html=True)
            
            # Apply styling
            st.dataframe(style_dataframe(
                problem_children[['ProductId', 'AverageScore', 'ReviewCount']].sort_values(by='AverageScore').head(items_to_display_in_table),
                header_bg_color=category_colors['Problem Child (Low Score, High Volume)'],
                header_text_color='white'
            ), hide_index=True, use_container_width=True, height=dataframe_height)
            
            st.markdown(f"""
            <div style="background-color: {category_colors['Problem Child (Low Score, High Volume)']}15; border-left: 3px solid {category_colors['Problem Child (Low Score, High Volume)']}; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: {insight_font_size};">
            <b>These popular items have lower satisfaction.</b>
            <b>Tips:</b>
            <ul>
                <li><b>Investigate Feedback:</b> Uncover common complaints to prioritize product improvements.</li>
                <li><b>Clarify Expectations:</b> Revamp product descriptions and images for accurate customer understanding.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            # --- Underperformers ---
            underperformers = product_performance[product_performance['Category'] == 'Underperformer (Low Score, Low Volume)']
            st.markdown(f"""
            <h3 style="color: {category_colors['Underperformer (Low Score, Low Volume)']}; font-size: 2em; text-align: left; margin-bottom: 0px;">
                {len(underperformers)} 'Underperformers'
            </h3>
            <p style="font-size: 1.2em; margin-top: 0px; color: #555;"><b>Low Review Volume, Low Average Score</b></p>
            """, unsafe_allow_html=True)

            # Apply styling
            st.dataframe(style_dataframe(
                underperformers[['ProductId', 'AverageScore', 'ReviewCount']].sort_values(by='ReviewCount').head(items_to_display_in_table),
                header_bg_color=category_colors['Underperformer (Low Score, Low Volume)'],
                header_text_color='white'
            ), hide_index=True, use_container_width=True, height=dataframe_height)
            
            st.markdown(f"""
            <div style="background-color: {category_colors['Underperformer (Low Score, Low Volume)']}15; border-left: 3px solid {category_colors['Underperformer (Low Score, Low Volume)']}; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: {insight_font_size};">
            <b>Products with limited engagement and satisfaction.</b>
            <b>Tips:</b>
            <ul>
                <li><b>Strategic Re-evaluation:</b> Assess market demand and competitive landscape; consider discontinuing if viability is low.</li>
                <li><b>Inventory Review:</b> Plan for clearance or bundling strategies to manage stock efficiently.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

        with col_right:
            st.write("#### Products for Growth & Maintenance")

            # --- Cash Cows ---
            cash_cows = product_performance[product_performance['Category'] == 'Cash Cow (High Score, High Volume)']
            st.markdown(f"""
            <h3 style="color: {category_colors['Cash Cow (High Score, High Volume)']}; font-size: 2em; text-align: left; margin-bottom: 0px;">
                {len(cash_cows)} 'Cash Cows'
            </h3>
            <p style="font-size: 1.2em; margin-top: 0px; color: #555;"><b>High Review Volume, High Average Score</b></p>
            """, unsafe_allow_html=True)
            
            # Apply styling
            st.dataframe(style_dataframe(
                cash_cows[['ProductId', 'AverageScore', 'ReviewCount']].sort_values(by='ReviewCount', ascending=False).head(items_to_display_in_table),
                header_bg_color=category_colors['Cash Cow (High Score, High Volume)'],
                header_text_color='white'
            ), hide_index=True, use_container_width=True, height=dataframe_height)
            
            st.markdown(f"""
            <div style="background-color: {category_colors['Cash Cow (High Score, High Volume)']}15; border-left: 3px solid {category_colors['Cash Cow (High Score, High Volume)']}; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: {insight_font_size};">
            <b>Strong performers in both quality and popularity.</b>
            <b>Tips:</b>
            <ul>
                <li><b>Maintain Quality & Visibility:</b> Uphold strict quality control and feature prominently in all marketing efforts.</li>
                <li><b>Cross-Sell/Upsell:</b> Leverage their popularity to introduce complementary or higher-value products.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            # --- Hidden Gems ---
            hidden_gems = product_performance[product_performance['Category'] == 'Hidden Gem (High Score, Low Volume)']
            st.markdown(f"""
            <h3 style="color: {category_colors['Hidden Gem (High Score, Low Volume)']}; font-size: 2em; text-align: left; margin-bottom: 0px;">
                {len(hidden_gems)} 'Hidden Gems'
            </h3>
            <p style="font-size: 1.2em; margin-top: 0px; color: #555;"><b>Low Review Volume, High Average Score</b></p>
            """, unsafe_allow_html=True)
            
            # Apply styling
            st.dataframe(style_dataframe(
                hidden_gems[['ProductId', 'AverageScore', 'ReviewCount']].sort_values(by='ReviewCount', ascending=False).head(items_to_display_in_table),
                header_bg_color=category_colors['Hidden Gem (High Score, Low Volume)'],
                header_text_color='white'
            ), hide_index=True, use_container_width=True, height=dataframe_height)
            
            st.markdown(f"""
            <div style="background-color: {category_colors['Hidden Gem (High Score, Low Volume)']}15; border-left: 3px solid {category_colors['Hidden Gem (High Score, Low Volume)']}; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: {insight_font_size};">
            <b>High satisfaction but limited exposure.</b>
            <b>Tips:</b>
            <ul>
                <li><b>Targeted Marketing & Promotions:</b> Increase awareness through focused campaigns, influencer outreach, or special offers.</li>
                <li><b>Encourage Reviews:</b> Actively seek more reviews to build social proof and boost visibility.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("---") # Separator before the new section

        # --- New Section: Individual Product Lookup ---
        st.write("### 🔍 Individual Product Lookup")
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <div data-testid="stInfo" style="background-color: #f0f2f6; border-left: 5px solid #6c757d; padding: 10px; border-radius: 5px; color: #333; font-size: 0.9em;">
                Select a product ID to view its performance metrics and classification category.
            </div>
        </div>
        """, unsafe_allow_html=True)

        all_product_ids = product_performance['ProductId'].unique().tolist()
        selected_product = st.selectbox("Select a Product ID:", [''] + sorted(all_product_ids))

        if selected_product:
            product_info = product_performance[product_performance['ProductId'] == selected_product].iloc[0]
            
            category = product_info['Category']
            avg_score = product_info['AverageScore']
            review_count = product_info['ReviewCount']
            
            # Extract category name and detailed description
            category_name = category.split(' (')[0] # e.g., 'Cash Cow'
            category_desc = category.split(' (')[1][:-1] # e.g., 'High Score, High Volume'

            st.markdown(f"#### Details for Product: <span style='color: #007BFF;'>{selected_product}</span>", unsafe_allow_html=True)
            
            col_detail1, col_detail2, col_detail3 = st.columns(3)

            with col_detail1:
                st.markdown(f"""
                <div style="background-color: #e6f7ff; border-left: 4px solid #007bff; padding: 10px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; font-size: 0.9em; color: #333;">Average Rating</p>
                    <h2 style="margin: 5px 0 0; color: #007bff;">{avg_score:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_detail2:
                st.markdown(f"""
                <div style="background-color: #e6f7ff; border-left: 4px solid #007bff; padding: 10px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; font-size: 0.9em; color: #333;">Total Reviews</p>
                    <h2 style="margin: 5px 0 0; color: #007bff;">{int(review_count):,}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col_detail3:
                # Use the category color for the background border
                cat_color = category_colors.get(category, '#6c757d') # Default to gray if not found
                st.markdown(f"""
                <div style="background-color: {cat_color}15; border-left: 4px solid {cat_color}; padding: 10px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; font-size: 0.9em; color: #333;">Category</p>
                    <h2 style="margin: 5px 0 0; color: {cat_color};">{category_name}</h2>
                    <p style="margin: 0; font-size: 0.8em; color: #555;">{category_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---") # Another separator below product details