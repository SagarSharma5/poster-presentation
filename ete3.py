import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from faker import Faker
import random

# Set page configuration
st.set_page_config(
    page_title="National Poster Presentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4500; /* Bright orange-red */
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #FF8C00, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333333; /* Dark gray for better readability */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #FF8C00;
        padding-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .card {
        padding: 1rem;
        border-radius: 1.5rem;
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        margin-bottom: 1rem;
        border-left: 4px solid #FF4500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 2rem;
        background: linear-gradient(135deg, #FFC107, #FF9800); /* Bright yellow to orange gradient */
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #333333;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111111;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    .stApp {
        background: linear-gradient(120deg, #E0FFFF, #F0F8FF); /* Light cyan to light blue gradient */
        color: #0D47A1;
    }
    
    /* Sidebar styling with vibrant gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #8E2DE2, #4A00E0); /* Vibrant purple gradient */
        border-right: 2px solid #4A00E0;
    }
    
    /* Sidebar title text color with enhanced styling */
    [data-testid="stSidebar"] h2 {
        color: white;
        text-align: center;
        padding: 1rem 0;
        font-weight: 600;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar text and controls styling */
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #EEEEEE; /* Light gray for better contrast against dark background */
        font-weight: 500;
    }
    
    /* Enhance sidebar filter section */
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF; /* Pure white for better visibility */
        font-size: 1.3rem;
        margin-top: 1.5rem;
        border-left: 3px solid #FFC107;
        padding-left: 0.5rem;
    }
    
    /* Make buttons more vibrant */
    .stButton>button {
        background-color: #FF4500 !important;
        color: white !important;
        font-weight: 500 !important;
        border-radius: 0.5rem !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #FF8C00 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Enhance dataframe styling */
    .stDataFrame {
        border-radius: 1.2rem !important;
        overflow: hidden !important;
        border: 1px solid #FF8C00 !important;
    }
    
    /* Enhance expander styling */
    .streamlit-expanderHeader {
        background-color: #E3F2FD !important;
        border-radius: 1.2rem !important;
        border-left: 4px solid #FF4500 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate dataset if it doesn't exist
def generate_event_dataset(num_participants=400):
    # Define constants
    tracks = ['Engineering Sciences', 'Life Sciences', 'Social Sciences', 'Physical Sciences']
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
    states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat', 'Telangana', 
              'West Bengal', 'Uttar Pradesh', 'Rajasthan', 'Punjab', 'Kerala', 'Haryana']
    colleges = [
        'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
        'NIT Trichy', 'NIT Warangal', 'NIT Surathkal', 'BITS Pilani', 'BITS Hyderabad',
        'Delhi University', 'Mumbai University', 'Pune University', 'Anna University',
        'Bangalore University', 'Jadavpur University', 'Amity University', 'VIT Vellore',
        'IIIT Hyderabad', 'SRM Chennai'
    ]
    
    # Feedback templates
    positive_templates = [
        "Excellent presentation on {topic}. The research methodology was {adjective}.",
        "Very informative poster about {topic}. The results were {adjective}.",
        "Impressive work on {topic}. The analysis was {adjective} and thorough.",
        "Great insights into {topic}. The visual representation was {adjective}.",
        "Fascinating research on {topic}. The conclusions were {adjective}."
    ]
    
    neutral_templates = [
        "Interesting approach to {topic}. The methodology could be {adjective}.",
        "Good presentation on {topic}, though the data could be {adjective}.",
        "Nice effort on {topic}, but the analysis needs to be more {adjective}.",
        "Decent work on {topic}, would benefit from more {adjective} examples.",
        "Standard research on {topic}, could use more {adjective} perspectives."
    ]
    
    critical_templates = [
        "The presentation on {topic} lacked {adjective} evidence.",
        "The poster about {topic} needs more {adjective} analysis.",
        "Research on {topic} requires more {adjective} methodology.",
        "The work on {topic} could benefit from {adjective} references.",
        "The study on {topic} needs {adjective} clarification."
    ]
    
    topics = {
        'Engineering Sciences': ['Renewable Energy', 'Machine Learning', 'IoT Applications', 'Robotics', 'Sustainable Architecture'],
        'Life Sciences': ['Genetic Engineering', 'Microbiology', 'Neuroscience', 'Biotechnology', 'Ecology'],
        'Social Sciences': ['Urban Development', 'Economic Policy', 'Cultural Studies', 'Educational Reform', 'Behavioral Psychology'],
        'Physical Sciences': ['Quantum Physics', 'Materials Science', 'Astronomical Observations', 'Climate Modeling', 'Chemical Synthesis']
    }
    
    positive_adjectives = ['innovative', 'comprehensive', 'detailed', 'rigorous', 'insightful']
    neutral_adjectives = ['clearer', 'more detailed', 'more structured', 'better organized', 'more focused']
    critical_adjectives = ['stronger', 'more coherent', 'better designed', 'more robust', 'more comprehensive']
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    fake = Faker()
    Faker.seed(42)
    
    # Initialize an empty list to store the data
    data = []
    
    for i in range(num_participants):
        participant_id = f'P{i+1:03d}'
        name = fake.name()
        track = random.choice(tracks)
        day = random.choice(days)
        state = random.choice(states)
        college = random.choice(colleges)
        topic = random.choice(topics[track])
        
        # Generate scores
        content_score = round(random.uniform(5.0, 10.0), 1)
        design_score = round(random.uniform(5.0, 10.0), 1)
        presentation_score = round(random.uniform(5.0, 10.0), 1)
        overall_score = round((content_score + design_score + presentation_score) / 3, 1)
        
        # Generate feedback
        feedback_type = random.choices(['positive', 'neutral', 'critical'], weights=[0.5, 0.3, 0.2])[0]
        if feedback_type == 'positive':
            template = random.choice(positive_templates)
            adjective = random.choice(positive_adjectives)
        elif feedback_type == 'neutral':
            template = random.choice(neutral_templates)
            adjective = random.choice(neutral_adjectives)
        else:
            template = random.choice(critical_templates)
            adjective = random.choice(critical_adjectives)
        
        feedback = template.format(topic=topic, adjective=adjective)
        
        # Append the data
        data.append({
            'Participant_ID': participant_id,
            'Name': name,
            'Track': track,
            'Day': day,
            'State': state,
            'College': college,
            'Content_Score': content_score,
            'Design_Score': design_score,
            'Presentation_Score': presentation_score,
            'Overall_Score': overall_score,
            'Feedback': feedback,
            'Topic': topic
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to generate sample images for each track
def generate_sample_images():
    # Create a directory for images if it doesn't exist
    if not os.path.exists('track_images'):
        os.makedirs('track_images')
    
    # Dictionary to map tracks to image colors/themes
    track_colors = {
        'Engineering Sciences': (30, 144, 255),  # DodgerBlue
        'Life Sciences': (50, 205, 50),  # LimeGreen
        'Social Sciences': (255, 165, 0),  # Orange
        'Physical Sciences': (148, 0, 211)  # DarkViolet
    }
    
    track_images = {}
    
    # Generate a sample image for each track and day
    tracks = ['Engineering Sciences', 'Life Sciences', 'Social Sciences', 'Physical Sciences']
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
    
    for track in tracks:
        track_images[track] = {}
        color = track_colors[track]
        
        for day in days:
            # Create a colored image with text
            img = Image.new('RGB', (500, 350), color=color)
            
            # Save the image
            img_path = f"track_images/{track.replace(' ', '_')}_{day.replace(' ', '_')}.jpg"
            img.save(img_path)
            track_images[track][day] = img_path
    
    return track_images

# Load data (or generate if it doesn't exist)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('poster_presentation_data.csv')
    except FileNotFoundError:
        df = generate_event_dataset()
        df.to_csv('poster_presentation_data.csv', index=False)
    return df

# Load sample images
@st.cache_data
def load_images():
    try:
        # Check if track images directory exists and has files
        if not os.path.exists('track_images') or len(os.listdir('track_images')) < 16:
            return generate_sample_images()
        
        # Load existing images
        track_images = {}
        tracks = ['Engineering Sciences', 'Life Sciences', 'Social Sciences', 'Physical Sciences']
        days = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
        
        for track in tracks:
            track_images[track] = {}
            for day in days:
                img_path = f"track_images/{track.replace(' ', '_')}_{day.replace(' ', '_')}.jpg"
                if os.path.exists(img_path):
                    track_images[track][day] = img_path
        
        return track_images
    except:
        return generate_sample_images()

# Load the data
df = load_data()
track_images = load_images()

# Sidebar
st.sidebar.markdown("<h2 style='text-align: center;'>Dashboard Controls</h2>", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview Dashboard", "Track Analysis", "Text Analysis", "Image Gallery"]
)

# Add global filters to sidebar
st.sidebar.markdown("### Filters")

# Track filter
selected_track = st.sidebar.multiselect(
    "Select Track(s)",
    options=df['Track'].unique(),
    default=df['Track'].unique()
)

# Day filter
selected_day = st.sidebar.multiselect(
    "Select Day(s)",
    options=df['Day'].unique(),
    default=df['Day'].unique()
)

# State filter
selected_state = st.sidebar.multiselect(
    "Select State(s)",
    options=df['State'].unique(),
    default=[]  # Default to no filter
)

# College filter
selected_college = st.sidebar.multiselect(
    "Select College(s)",
    options=df['College'].unique(),
    default=[]  # Default to no filter
)

# Apply filters
filtered_df = df.copy()

if selected_track:
    filtered_df = filtered_df[filtered_df['Track'].isin(selected_track)]
if selected_day:
    filtered_df = filtered_df[filtered_df['Day'].isin(selected_day)]
if selected_state:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_state)]
if selected_college:
    filtered_df = filtered_df[filtered_df['College'].isin(selected_college)]

# Overview Dashboard
if page == "Overview Dashboard":
    st.markdown("<div class='main-header'>National Poster Presentation Event Dashboard</div>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(filtered_df)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Participants</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        avg_score = filtered_df['Overall_Score'].mean()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_score:.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Average Score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        top_track = filtered_df.groupby('Track').size().idxmax()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{top_track}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Most Popular Track</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        top_college = filtered_df.groupby('College').size().idxmax()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{top_college}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Top College</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Participation Distribution</div>", unsafe_allow_html=True)
    
    # Visualizations Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Track-wise distribution
        fig_track = px.pie(
            filtered_df, 
            names='Track', 
            title='Participation by Track',
            color='Track',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_track.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111111'),
            legend=dict(font=dict(color='#111111'))
        )
        fig_track.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(color='#111111'))
        st.plotly_chart(fig_track, use_container_width=True)
    
    with col2:
        # Day-wise distribution
        fig_day = px.bar(
            filtered_df.groupby('Day').size().reset_index(name='count'),
            x='Day',
            y='count',
            color='Day',
            title='Participation by Day',
            labels={'count': 'Number of Participants'},
            color_discrete_sequence=['#FF4500', '#FFC107', '#4A00E0', '#00C853']
        )
        fig_day.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
            font=dict(color='#111111'),
            legend=dict(font=dict(color='#111111'))
        )
        st.plotly_chart(fig_day, use_container_width=True)
    
    # Visualizations Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Track performance box plot
        fig_scores = px.box(
            filtered_df,
            x='Track',
            y='Overall_Score',
            color='Track',
            title='Score Distribution by Track',
            points='all',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_scores.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
            font=dict(color='#111111'),
            legend=dict(font=dict(color='#111111'))
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        # State-wise distribution
        state_counts = filtered_df.groupby('State').size().reset_index(name='count')
        state_counts = state_counts.sort_values('count', ascending=False)
        
        fig_state = px.bar(
            state_counts,
            x='State',
            y='count',
            title='Participation by State',
            color='count',
            color_continuous_scale=['#4A00E0', '#8E2DE2', '#FF4500', '#FF8C00', '#FFC107'],
            labels={'count': 'Number of Participants'}
        )
        fig_state.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
            font=dict(color='#111111'),
            legend=dict(font=dict(color='#111111'))
        )
        fig_state.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_state, use_container_width=True)
    
    # Visualizations Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        # Day-Track heatmap
        day_track_counts = filtered_df.groupby(['Day', 'Track']).size().unstack(fill_value=0)
        fig_heatmap = px.imshow(
            day_track_counts,
            labels=dict(x="Track", y="Day", color="Count"),
            title="Participation Heatmap: Day vs Track",
            color_continuous_scale=['#4A00E0', '#8E2DE2', '#FF4500', '#FF8C00', '#FFC107']
        )
        fig_heatmap.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111111')
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Top 10 colleges
        college_counts = filtered_df.groupby('College').size().reset_index(name='count')
        college_counts = college_counts.sort_values('count', ascending=False).head(10)
        
        fig_college = px.bar(
            college_counts,
            x='count',
            y='College',
            title='Top 10 Colleges by Participation',
            orientation='h',
            color='count',
            color_continuous_scale=['#4A00E0', '#8E2DE2', '#FF4500', '#FF8C00', '#FFC107'],
            labels={'count': 'Number of Participants'}
        )
        fig_college.update_layout(
            title_font=dict(size=20, color='#111111'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
            yaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
            font=dict(color='#111111')
        )
        st.plotly_chart(fig_college, use_container_width=True)
    
    # Raw Data View (Collapsible)
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)

# Track Analysis
elif page == "Track Analysis":
    st.markdown("<div class='main-header'>Track Analysis</div>", unsafe_allow_html=True)
    
    if not selected_track:
        st.warning("Please select at least one track in the sidebar.")
    else:
        # Key metrics for each track
        track_metrics = filtered_df.groupby('Track').agg({
            'Participant_ID': 'count',
            'Overall_Score': 'mean',
            'Content_Score': 'mean',
            'Design_Score': 'mean',
            'Presentation_Score': 'mean'
        }).reset_index()
        
        for _, row in track_metrics.iterrows():
            if row['Track'] in selected_track:
                st.markdown(f"<div class='sub-header'>{row['Track']}</div>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{int(row['Participant_ID'])}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Participants</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{row['Content_Score']:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Avg. Content Score</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{row['Design_Score']:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Avg. Design Score</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{row['Presentation_Score']:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Avg. Presentation Score</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                track_df = filtered_df[filtered_df['Track'] == row['Track']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Day-wise distribution for this track
                    track_day_counts = track_df.groupby('Day').size().reset_index(name='count')
                    fig_track_day = px.bar(
                        track_day_counts, 
                        x='Day', 
                        y='count',
                        title=f"Day-wise Participation for {row['Track']}",
                        color='Day',
                        labels={'count': 'Number of Participants'},
                        color_discrete_sequence=['#FF4500', '#FFC107', '#4A00E0', '#00C853']
                    )
                    fig_track_day.update_layout(
                        title_font=dict(size=20, color='#111111'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
                        font=dict(color='#111111')
                    )
                    st.plotly_chart(fig_track_day, use_container_width=True)
                
                with col2:
                    # Score comparison
                    scores_df = track_df[['Content_Score', 'Design_Score', 'Presentation_Score']]
                    scores_melt = pd.melt(scores_df, var_name='Score Type', value_name='Score')
                    
                    fig_scores = px.violin(
                        scores_melt,
                        y='Score',
                        x='Score Type',
                        box=True,
                        title=f"Score Distribution for {row['Track']}",
                        color='Score Type',
                        color_discrete_sequence=['#FF4500', '#FFC107', '#4A00E0']
                    )
                    fig_scores.update_layout(
                        title_font=dict(size=20, color='#111111'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
                        font=dict(color='#111111')
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                # Topic distribution within track
                topics = track_df['Topic'].value_counts().reset_index()
                topics.columns = ['Topic', 'Count']
                
                fig_topics = px.pie(
                    topics,
                    values='Count',
                    names='Topic',
                    title=f"Topic Distribution in {row['Track']}",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_topics.update_layout(
                    title_font=dict(size=20, color='#111111'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#111111')
                )
                fig_topics.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(color='#111111'))
                st.plotly_chart(fig_topics, use_container_width=True)
                
                # Show top performers
                st.markdown("<div class='sub-header'>Top Performers</div>", unsafe_allow_html=True)
                top_performers = track_df.sort_values('Overall_Score', ascending=False).head(5)
                st.dataframe(top_performers[['Participant_ID', 'Name', 'College', 'State', 'Overall_Score']])

# Text Analysis
elif page == "Text Analysis":
    st.markdown("<div class='main-header'>Feedback Text Analysis</div>", unsafe_allow_html=True)
    
    # Track selector for text analysis
    text_track = st.selectbox(
        "Select a Track for Feedback Analysis",
        options=df['Track'].unique()
    )
    
    track_feedback = filtered_df[filtered_df['Track'] == text_track]['Feedback'].tolist()
    
    if not track_feedback:
        st.warning(f"No feedback data available for {text_track} with the current filters.")
    else:
        st.markdown("<div class='sub-header'>Feedback Word Cloud</div>", unsafe_allow_html=True)
        
        # Generate and display word cloud
        all_feedback = ' '.join(track_feedback)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            colormap='viridis',
            max_words=100,
            collocations=False
        ).generate(all_feedback)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Text similarity analysis
        st.markdown("<div class='sub-header'>Feedback Similarity Analysis</div>", unsafe_allow_html=True)
        
        if len(track_feedback) < 2:
            st.warning("Need at least two feedback entries to perform similarity analysis.")
        else:
            # Vectorize the feedback text
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(track_feedback)
            
            # Compute similarity between all pairs
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Plot similarity heatmap
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix[:min(len(similarity_matrix), 20), :min(len(similarity_matrix), 20)], 
                        annot=False, cmap='viridis')
            plt.title(f'Feedback Similarity Matrix for {text_track}')
            st.pyplot(fig)
            
            st.write("The heatmap shows similarity between feedback entries. Brighter colors indicate higher similarity.")
            
            # Most common themes
            st.markdown("<div class='sub-header'>Common Feedback Themes</div>", unsafe_allow_html=True)
            
            # Get the top terms from TF-IDF
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Sum TF-IDF values for each term across all documents
            tfidf_sums = tfidf_matrix.sum(axis=0).A1
            
            # Get the indices of top terms
            top_indices = tfidf_sums.argsort()[-10:][::-1]
            
            # Get the top terms and their scores
            top_terms = [(feature_names[i], tfidf_sums[i]) for i in top_indices]
            
            # Create a bar chart
            terms_df = pd.DataFrame(top_terms, columns=['Term', 'Importance'])
            
            fig_terms = px.bar(
                terms_df,
                x='Importance',
                y='Term',
                orientation='h',
                title="Top Terms in Feedback",
                color='Importance',
                color_continuous_scale=['#4A00E0', '#8E2DE2', '#FF4500', '#FF8C00', '#FFC107']
            )
            fig_terms.update_layout(
                title_font=dict(size=20, color='#111111'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#111111')),
                yaxis=dict(showgrid=False, tickfont=dict(color='#111111')),
                font=dict(color='#111111')
            )
            st.plotly_chart(fig_terms, use_container_width=True)
            
            # Display sample feedback
            st.markdown("<div class='sub-header'>Sample Feedback</div>", unsafe_allow_html=True)
            sample_feedback = filtered_df[filtered_df['Track'] == text_track].sample(min(5, len(track_feedback)))
            for idx, row in sample_feedback.iterrows():
                st.write(f"**Participant:** {row['Name']} (ID: {row['Participant_ID']})")
                st.write(f"**Topic:** {row['Topic']}")
                st.write(f"**Feedback:** {row['Feedback']}")
                st.write("---")

# Image Gallery
elif page == "Image Gallery":
    st.markdown("<div class='main-header'>Image Gallery & Processing</div>", unsafe_allow_html=True)
    
    # Day selector for image gallery
    img_day = st.selectbox(
        "Select a Day",
        options=df['Day'].unique()
    )
    
    # Display images by track for the selected day
    st.markdown("<div class='sub-header'>Track Images</div>", unsafe_allow_html=True)
    st.write(f"Displaying images for {img_day}")
    
    col1, col2 = st.columns(2)
    
    tracks = df['Track'].unique()
    for i, track in enumerate(tracks):
        img_path = track_images[track][img_day]
        img = Image.open(img_path)
        
        # Display in alternating columns
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"##### {track}")
            st.image(img, caption=f"{track} - {img_day}", use_column_width=True)
    
    # Image processing section
    st.markdown("<div class='sub-header'>Image Processing</div>", unsafe_allow_html=True)
    
    # Track selector for image processing
    process_track = st.selectbox(
        "Select Track for Image Processing",
        options=df['Track'].unique()
    )
    
    # Get image for the selected track and day
    img_path = track_images[process_track][img_day]
    img = Image.open(img_path)
    
    # Show original image
    st.markdown("##### Original Image")
    st.image(img, caption=f"Original - {process_track} - {img_day}", use_column_width=True)
    
    # Image processing options
    st.markdown("##### Select Processing Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_option = st.selectbox(
            "Apply Filter",
            ["None", "Blur", "Contour", "Edge Enhance", "Emboss", "Sharpen"]
        )
    
    with col2:
        color_option = st.selectbox(
            "Color Adjustment",
            ["None", "Grayscale", "Sepia", "Invert", "Enhance Color", "Enhance Contrast"]
        )
    
    with col3:
        transform_option = st.selectbox(
            "Transform",
            ["None", "Rotate 90°", "Rotate 180°", "Rotate 270°", "Flip Horizontal", "Flip Vertical"]
        )
    
    # Process the image
    processed_img = img.copy()
    
    # Apply filter
    if filter_option == "Blur":
        processed_img = processed_img.filter(ImageFilter.BLUR)
    elif filter_option == "Contour":
        processed_img = processed_img.filter(ImageFilter.CONTOUR)
    elif filter_option == "Edge Enhance":
        processed_img = processed_img.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_option == "Emboss":
        processed_img = processed_img.filter(ImageFilter.EMBOSS)
    elif filter_option == "Sharpen":
        processed_img = processed_img.filter(ImageFilter.SHARPEN)
    
    # Apply color adjustment
    if color_option == "Grayscale":
        processed_img = processed_img.convert('L').convert('RGB')
    elif color_option == "Sepia":
        # Simple sepia filter
        gray_img = processed_img.convert('L')
        sepia_img = Image.merge('RGB', [
            gray_img.point(lambda x: min(255, int(x * 1.1))),
            gray_img.point(lambda x: min(255, int(x * 0.9))),
            gray_img.point(lambda x: min(255, int(x * 0.7)))
        ])
        processed_img = sepia_img
    elif color_option == "Invert":
        processed_img = ImageOps.invert(processed_img.convert('RGB'))
    elif color_option == "Enhance Color":
        enhancer = ImageEnhance.Color(processed_img)
        processed_img = enhancer.enhance(1.5)
    elif color_option == "Enhance Contrast":
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(1.5)
    
    # Apply transform
    if transform_option == "Rotate 90°":
        processed_img = processed_img.rotate(90, expand=True)
    elif transform_option == "Rotate 180°":
        processed_img = processed_img.rotate(180)
    elif transform_option == "Rotate 270°":
        processed_img = processed_img.rotate(270, expand=True)
    elif transform_option == "Flip Horizontal":
        processed_img = processed_img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_option == "Flip Vertical":
        processed_img = processed_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Show processed image
    st.markdown("##### Processed Image")
    st.image(processed_img, caption=f"Processed - {process_track} - {img_day}", use_column_width=True)
    
    # Option to download processed image
    buffered = io.BytesIO()
    processed_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="processed_{process_track}_{img_day}.jpg">Download Processed Image</a>'
    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>National Poster Presentation Event Dashboard | Created with Streamlit</p>", unsafe_allow_html=True)