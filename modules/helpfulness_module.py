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
            
            # Additional insight for distribution plot
            if not df_voted.empty:
                zero_helpful_ratio_count = (df_voted['HelpfulnessRatio'] == 0).sum()
                if len(df_voted) > 0:
                    zero_helpful_percentage = (zero_helpful_ratio_count / len(df_voted)) * 100
                    st.markdown(f"**Analysis:** Approximately **{zero_helpful_percentage:.1f}%** of all voted reviews received zero helpful votes, indicating they weren't found useful by anyone who voted.")
                else:
                    st.markdown("**Analysis:** No voted reviews available for this analysis.")
            else:
                st.markdown("**Analysis:** No voted reviews available for this analysis.")


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
            
            # Additional insight for score vs helpfulness
            if not avg_help_by_score.empty:
                max_help_score = avg_help_by_score.loc[avg_help_by_score['HelpfulnessRatio'].idxmax()]
                st.markdown(f"**Analysis:** Reviews with a **{int(max_help_score['Score'])}-star** rating tend to have the highest average helpfulness ratio of **{max_help_score['HelpfulnessRatio']:.1%}**.")
            else:
                st.markdown("**Analysis:** No data to show average helpfulness by score.")

        st.markdown("---")

        # --- NEW ROW: Helpfulness Trends Over Time & Top Helpful Users ---
        st.write("### ⏱️ Trends & Top Contributors")
        col_trend, col_top_users = st.columns([0.6, 0.4]) # Adjust column width ratio as needed

        with col_trend:
            st.write("#### 📈 Helpfulness Trends Over Time")
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
                                        height=400, # Adjusted height for side-by-side
                                        color_discrete_sequence=[px.colors.qualitative.Plotly[5]])
            fig_help_trend.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_help_trend, use_container_width=True)
            st.info("💡 **Insight:** Observe if review helpfulness is generally increasing or decreasing. A declining trend might suggest changes in community engagement or the quality of new reviews.")
            
            # Additional insight for helpfulness trend
            if not filtered_help_over_time.empty:
                start_avg = filtered_help_over_time['HelpfulnessRatio'].iloc[0]
                end_avg = filtered_help_over_time['HelpfulnessRatio'].iloc[-1]
                trend_direction = "increased" if end_avg > start_avg else "decreased" if end_avg < start_avg else "remained stable"
                st.markdown(f"**Analysis:** Over the selected period, the average helpfulness ratio has generally **{trend_direction}** from **{start_avg:.1%}** to **{end_avg:.1%}**.")
            else:
                st.markdown("**Analysis:** No data to show helpfulness trends over the selected period.")


        with col_top_users:
            st.write("#### 🏆 Top Helpful Reviewers")
            st.info("Identify users who consistently provide helpful reviews by the total number of helpful votes their reviews have received.")

            # Calculate total helpful votes per user
            user_helpfulness = df_voted.groupby('UserId').agg(
                total_helpful_votes=('HelpfulnessNumerator', 'sum'),
                total_reviews=('Id', 'count')
            ).reset_index()

            # Filter for users with a significant number of helpful votes to avoid noise
            min_helpful_votes_user = st.slider("Min Total Helpful Votes for a User:", 0, 500, 10, key='min_helpful_user_votes_sidebar')
            filtered_user_helpfulness = user_helpfulness[user_helpfulness['total_helpful_votes'] >= min_helpful_votes_user]

            # Sort by total helpful votes and display top users
            top_n_users_sidebar = st.slider("Number of Top Reviewers to Show:", 5, 50, 10, key='top_n_users_sidebar')
            top_helpful_users = filtered_user_helpfulness.sort_values(by='total_helpful_votes', ascending=False).head(top_n_users_sidebar)

            if not top_helpful_users.empty:
                st.dataframe(top_helpful_users.set_index('UserId').style.background_gradient(cmap='Greens', subset=['total_helpful_votes']), use_container_width=True, height=400) # Adjusted height
                st.info("💡 **Insight:** These users are gold! Their reviews are consistently found valuable by others. Understanding their review style can be beneficial.")
                
                # Additional insight for top helpful reviewers
                most_helpful_user = top_helpful_users.iloc[0]
                st.markdown(f"**Analysis:** The most helpful reviewer, **{most_helpful_user['UserId']}**, has contributed to **{int(most_helpful_user['total_helpful_votes'])}** helpful votes across **{int(most_helpful_user['total_reviews'])}** reviews.")

            else:
                st.info("No top reviewers available based on the selected minimum helpful votes.")

        st.markdown("---")
        
        # --- Impactful Reviews: Learn from the Best & Worst (Original Row 2) ---
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