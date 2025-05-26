import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data_helpfulness():
    try:
        df = pd.read_csv('Reviews.csv')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce') # Ensure Time is datetime

        # Handle HelpfulnessRatio: if denominator is 0, ratio is 0
        df['HelpfulnessRatio'] = df.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
        # Filter out reviews where HelpfulnessDenominator is 0 for analysis that relies on votes
        df_voted = df[df['HelpfulnessDenominator'] > 0].copy()
        
        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)
        return df, df_voted # Return both original and voted-on dataframe
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data: {e}")
        st.stop()

def show_helpfulness_dashboard():
    st.title("👍 Helpfulness Deep Dive: What Makes Reviews Impactful?")
    st.markdown("---") # Visual separator

    df, df_voted = load_data_helpfulness()

    if df is not None:
        # --- Row 1: Distribution of Helpfulness Ratio & Score vs. Helpfulness ---
        st.write("### 📊 Helpfulness Dynamics")
        col_h_plot1, col_h_plot2 = st.columns(2)

        with col_h_plot1:
            st.write("#### Distribution of Helpfulness Ratio")
            st.info("The **'Helpfulness Ratio'** shows the proportion of users who found a review helpful. A higher value indicates more impact. The X-axis is fixed from 0.0 to 1.0.")
            
            fig_helpfulness_dist = px.histogram(df_voted, x='HelpfulnessRatio',
                                                nbins=50,
                                                title='Distribution of Helpful Reviews',
                                                labels={'HelpfulnessRatio': 'Helpfulness Ratio'},
                                                color_discrete_sequence=[px.colors.sequential.Viridis[3]], # A nice green-blue
                                                height=400) # Consistent height
            fig_helpfulness_dist.update_layout(xaxis_title="Helpfulness Ratio (Helpful Votes / Total Votes)",
                                              yaxis_title="Number of Reviews",
                                              xaxis_range=[0, 1], # Set X-axis range from 0 to 1
                                              margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_helpfulness_dist, use_container_width=True)
            st.info("💡 **Insight:** A significant peak at 0.0 means many reviews get no 'helpful' votes, while the tail towards 1.0 showcases highly impactful reviews. Aim to understand the latter!")

        with col_h_plot2:
            st.write("#### Average Helpfulness by Score")
            st.info("Does a higher star rating correlate with more helpfulness? Explore the trend for reviews that received votes.")
            
            avg_help_by_score = df_voted.groupby('Score')['HelpfulnessRatio'].mean().reset_index()

            fig_score_help = px.bar(avg_help_by_score,
                                    x='Score',
                                    y='HelpfulnessRatio',
                                    title='Average Helpfulness Ratio by Star Rating',
                                    labels={'Score': 'Star Rating', 'HelpfulnessRatio': 'Average Helpfulness Ratio'},
                                    color='HelpfulnessRatio',
                                    color_continuous_scale=px.colors.sequential.Plasma, # Different color scale
                                    range_y=[0, 1], # Keep Y-axis fixed
                                    height=400) # Consistent height
            fig_score_help.update_layout(xaxis_title="Star Rating",
                                         yaxis_title="Average Helpfulness Ratio",
                                         margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_score_help, use_container_width=True)
            st.info("💡 **Insight:** This plot reveals if higher (or lower) rated reviews are perceived as more helpful. There might be a sweet spot, or specific scores could be more informative.")

        st.markdown("---")

        # --- Row 2: Most Helpful Reviews & Least Helpful Reviews (Tabbed Interface) ---
        st.write("### 🌟 Impactful Reviews: Learn from the Best & Worst")
        st.info("Dive into actual review examples. Use the sliders to define what 'most' or 'least' helpful means to you!")

        # Dynamic number of reviews to display, applies to both tabs
        num_reviews_display = st.slider(
            "Number of Reviews to Show per Tab:",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="Adjust to see more or fewer example reviews in each tab."
        )

        tab1, tab2 = st.tabs(["⭐ Most Helpful Reviews", "🗑️ Least Helpful Reviews"])

        with tab1:
            st.write("#### Criteria for Most Helpful Reviews:")
            col_criteria1, col_criteria2 = st.columns(2)
            with col_criteria1:
                helpful_threshold_tab1 = st.slider("Min Helpfulness Ratio:", 0.0, 1.0, 0.8, key='mhr_tab1', help="Only show reviews with this ratio or higher.")
            with col_criteria2:
                min_votes_tab1 = st.slider("Min Total Votes:", 0, 100, 5, key='mtv_tab1', help="Only show reviews with at least this many votes.")

            most_helpful_reviews = df_voted[
                (df_voted['HelpfulnessRatio'] >= helpful_threshold_tab1) &
                (df_voted['HelpfulnessDenominator'] >= min_votes_tab1)
            ].sort_values(by=['HelpfulnessRatio', 'HelpfulnessDenominator'], ascending=[False, False]).head(num_reviews_display)

            if not most_helpful_reviews.empty:
                st.write(f"Displaying top {len(most_helpful_reviews)} reviews matching your criteria:")
                for i, row in most_helpful_reviews.iterrows():
                    summary_text = row['Summary'] if row['Summary'] else "No Summary"
                    review_display_text = f"**Score:** {int(row['Score'])}/5 | **Helpful:** {int(row['HelpfulnessNumerator'])}/{int(row['HelpfulnessDenominator'])} ({row['HelpfulnessRatio']:.1%})"
                    
                    with st.expander(f"**{summary_text[:60]}...** - {review_display_text}"): # Truncate summary for expander title
                        st.write(f"**Product ID:** {row['ProductId']}")
                        st.write(f"**User ID:** {row['UserId']}")
                        st.write(f"**Time:** {row['Time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Time']) else 'N/A'}")
                        st.markdown(f"**Full Review Text:**\n```\n{row['Text']}\n```")
            else:
                st.info("No reviews match the selected criteria for 'Most Helpful'. Try adjusting the sliders above.")

        with tab2:
            st.write("#### Criteria for Least Helpful Reviews:")
            st.warning("These reviews received votes but were found **not helpful (Helpfulness Ratio = 0)**. How many votes did they receive?")
            min_votes_tab2 = st.slider("Min Total Votes for Least Helpful:", 0, 100, 5, key='mtv_tab2', help="Only show reviews with 0 helpfulness and at least this many total votes.")

            least_helpful_reviews = df_voted[
                (df_voted['HelpfulnessRatio'] == 0) &
                (df_voted['HelpfulnessDenominator'] >= min_votes_tab2)
            ].sort_values(by='HelpfulnessDenominator', ascending=False).head(num_reviews_display)

            if not least_helpful_reviews.empty:
                st.write(f"Displaying top {len(least_helpful_reviews)} reviews matching your criteria:")
                for i, row in least_helpful_reviews.iterrows():
                    summary_text = row['Summary'] if row['Summary'] else "No Summary"
                    review_display_text = f"**Score:** {int(row['Score'])}/5 | **Helpful:** {int(row['HelpfulnessNumerator'])}/{int(row['HelpfulnessDenominator'])} ({row['HelpfulnessRatio']:.1%})"
                    
                    with st.expander(f"**{summary_text[:60]}...** - {review_display_text}"): # Truncate summary for expander title
                        st.write(f"**Product ID:** {row['ProductId']}")
                        st.write(f"**User ID:** {row['UserId']}")
                        st.write(f"**Time:** {row['Time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Time']) else 'N/A'}")
                        st.markdown(f"**Full Review Text:**\n```\n{row['Text']}\n```")
            else:
                st.info("No reviews found that were voted on but received 0 helpfulness under your criteria. Try adjusting the 'Min Total Votes' slider.")

        st.markdown("---")
        st.write("### 📈 Helpfulness Trends Over Time")
        st.info("Is the average helpfulness of reviews changing over time? This can indicate shifts in review quality or community engagement.")

        df_time_filtered = df.dropna(subset=['Time'])
        df_time_filtered['YearMonth'] = df_time_filtered['Time'].dt.to_period('M').astype(str)

        # Only average helpfulness for reviews that received votes
        avg_help_over_time = df_time_filtered[df_time_filtered['HelpfulnessDenominator'] > 0].groupby('YearMonth')['HelpfulnessRatio'].mean().reset_index()
        avg_help_over_time['Date'] = pd.to_datetime(avg_help_over_time['YearMonth'])
        avg_help_over_time.sort_values('Date', inplace=True)

        time_filter_options = ["All Time", "Last 1 Year", "Last 3 Years", "Last 5 Years"]
        selected_time_filter = st.selectbox("Filter Helpfulness Trend:", time_filter_options, key='helpfulness_time_filter')

        filtered_help_over_time = avg_help_over_time.copy()
        if selected_time_filter == "Last 1 Year":
            filtered_help_over_time = avg_help_over_time[avg_help_over_time['Date'] >= (pd.to_datetime(avg_help_over_time['Date'].max()) - pd.DateOffset(years=1))]
        elif selected_time_filter == "Last 3 Years":
            filtered_help_over_time = avg_help_over_time[avg_help_over_time['Date'] >= (pd.to_datetime(avg_help_over_time['Date'].max()) - pd.DateOffset(years=3))]
        elif selected_time_filter == "Last 5 Years":
            filtered_help_over_time = avg_help_over_time[avg_help_over_time['Date'] >= (pd.to_datetime(avg_help_over_time['Date'].max()) - pd.DateOffset(years=5))]

        fig_help_trend = px.line(filtered_help_over_time,
                                 x='Date',
                                 y='HelpfulnessRatio',
                                 title='Average Review Helpfulness Over Time',
                                 labels={'Date': 'Date', 'HelpfulnessRatio': 'Average Helpfulness Ratio'},
                                 range_y=[0, 1],
                                 height=450,
                                 color_discrete_sequence=[px.colors.qualitative.Plotly[5]])
        fig_help_trend.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_help_trend, use_container_width=True)
        st.info("💡 **Insight:** Observe if review helpfulness is generally increasing or decreasing. A declining trend might suggest changes in community engagement or the quality of new reviews.")