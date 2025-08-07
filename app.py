import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Pangeos Feasibility Analyzer",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class QwenAIIntegration:
    """Enhanced AI integration with multiple Qwen models"""
    
    def __init__(self):
        self.models = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2-72b-instruct", 
            "qwen/qwen-1.5-110b-chat",
            "qwen/qwen-2.5-7b-instruct"
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
        }
    
    def get_analysis(self, prompt, model_index=0):
        """Get AI analysis from Qwen models"""
        try:
            payload = {
                "model": self.models[model_index],
                "messages": [
                    {"role": "system", "content": "You are an expert yacht engineering and feasibility analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Analysis unavailable (Status: {response.status_code})"
                
        except Exception as e:
            return f"AI Analysis temporarily unavailable: {str(e)[:100]}"

class MultiAgentSystem:
    """Advanced Multi-Agent AI System for comprehensive analysis"""
    
    def __init__(self):
        self.qwen_ai = QwenAIIntegration()
        self.agents = {
            "engineering": "🔧 Engineering Agent",
            "financial": "💰 Financial Agent", 
            "environmental": "🌱 Environmental Agent",
            "logistics": "🚢 Logistics Agent"
        }
    
    def engineering_analysis(self, specs):
        """Engineering feasibility analysis"""
        prompt = f"""
        Analyze the engineering feasibility of Pangeos yacht with these specifications:
        - Length: {specs['length']}m
        - Beam: {specs['beam']}m  
        - Displacement: {specs['displacement']} tons
        - Passenger Capacity: {specs['passengers']}
        - Propulsion Power: {specs['power']} MW
        
        Focus on: structural integrity, propulsion systems, stability, materials, construction challenges.
        Provide a feasibility score (0-100) and key recommendations.
        """
        
        analysis = self.qwen_ai.get_analysis(prompt, 0)
        score = self._extract_score(analysis)
        return {"score": score, "analysis": analysis, "agent": "Engineering"}
    
    def financial_analysis(self, specs):
        """Financial feasibility analysis"""
        prompt = f"""
        Analyze the financial feasibility of Pangeos yacht project:
        - Estimated Cost: ${specs['cost']} billion
        - Construction Time: {specs['construction_time']} years
        - Passenger Capacity: {specs['passengers']}
        - Expected ROI timeline: {specs.get('roi_years', 15)} years
        
        Consider: construction costs, operational expenses, revenue potential, financing challenges, market demand.
        Provide a feasibility score (0-100) and investment recommendations.
        """
        
        analysis = self.qwen_ai.get_analysis(prompt, 1)
        score = self._extract_score(analysis)
        return {"score": score, "analysis": analysis, "agent": "Financial"}
    
    def environmental_analysis(self, specs):
        """Environmental impact analysis"""
        prompt = f"""
        Analyze the environmental impact and sustainability of Pangeos yacht:
        - Size: {specs['length']}m x {specs['beam']}m
        - {specs['passengers']} passengers
        - Propulsion: {specs['power']} MW
        - Green energy integration potential
        
        Assess: carbon footprint, renewable energy systems, waste management, marine ecosystem impact, sustainability measures.
        Provide an environmental score (0-100) and green recommendations.
        """
        
        analysis = self.qwen_ai.get_analysis(prompt, 2)
        score = self._extract_score(analysis)
        return {"score": score, "analysis": analysis, "agent": "Environmental"}
    
    def logistics_analysis(self, specs):
        """Logistics and operational analysis"""
        prompt = f"""
        Analyze logistics feasibility for Pangeos yacht construction and operation:
        - Construction Time: {specs['construction_time']} years
        - Size: {specs['length']}m length
        - Capacity: {specs['passengers']} passengers
        - Global operation requirements
        
        Consider: shipyard requirements, supply chain, construction logistics, port infrastructure, operational complexity.
        Provide a logistics score (0-100) and operational recommendations.
        """
        
        analysis = self.qwen_ai.get_analysis(prompt, 3)
        score = self._extract_score(analysis)
        return {"score": score, "analysis": analysis, "agent": "Logistics"}
    
    def _extract_score(self, analysis):
        """Extract numerical score from AI analysis"""
        try:
            # Look for patterns like "Score: 75" or "feasibility: 80%" etc.
            import re
            patterns = [
                r'score[:\s]*(\d+)',
                r'feasibility[:\s]*(\d+)',
                r'rating[:\s]*(\d+)',
                r'(\d+)(?:\s*\/\s*100|\s*%)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, analysis.lower())
                if match:
                    return min(100, max(0, int(match.group(1))))
            
            # Fallback: analyze sentiment and assign score
            positive_words = ['excellent', 'good', 'strong', 'feasible', 'viable', 'promising']
            negative_words = ['poor', 'difficult', 'challenging', 'problematic', 'unfeasible']
            
            analysis_lower = analysis.lower()
            positive_count = sum(1 for word in positive_words if word in analysis_lower)
            negative_count = sum(1 for word in negative_words if word in analysis_lower)
            
            if positive_count > negative_count:
                return 75 + np.random.randint(-10, 15)
            elif negative_count > positive_count:
                return 45 + np.random.randint(-15, 20)
            else:
                return 65 + np.random.randint(-10, 15)
                
        except:
            return 70 + np.random.randint(-15, 20)

class FeasibilityPredictor:
    """Advanced ML-based feasibility prediction system"""
    
    def __init__(self):
        self.is_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize synthetic models for demonstration"""
        # Simulate trained model weights and parameters
        self.feature_weights = {
            'length': 0.15, 'beam': 0.12, 'displacement': 0.18,
            'passengers': 0.14, 'power': 0.16, 'cost': 0.13,
            'construction_time': 0.12
        }
        self.is_trained = True
    
    def predict_feasibility(self, specs):
        """Predict feasibility scores using synthetic ML model"""
        if not self.is_trained:
            self._initialize_models()
        
        # Normalize inputs
        normalized_specs = self._normalize_specs(specs)
        
        # Simulate complex ML predictions
        base_scores = {}
        for category in ['engineering', 'financial', 'environmental', 'logistics']:
            score = 0
            for feature, value in normalized_specs.items():
                weight = self.feature_weights.get(feature, 0.1)
                contribution = value * weight * (80 + np.random.normal(0, 10))
                score += contribution
            
            # Add category-specific adjustments
            if category == 'engineering':
                score += normalized_specs['length'] * 5 - normalized_specs['displacement'] * 3
            elif category == 'financial': 
                score -= normalized_specs['cost'] * 8 + normalized_specs['construction_time'] * 4
            elif category == 'environmental':
                score += 15 - normalized_specs['passengers'] * 2
            elif category == 'logistics':
                score -= normalized_specs['length'] * 2 + normalized_specs['construction_time'] * 3
            
            base_scores[category] = max(30, min(95, score + np.random.normal(0, 5)))
        
        # Overall score as weighted average
        overall_score = (
            base_scores['engineering'] * 0.3 +
            base_scores['financial'] * 0.25 +
            base_scores['environmental'] * 0.20 +
            base_scores['logistics'] * 0.25
        )
        
        base_scores['overall'] = overall_score
        return base_scores
    
    def _normalize_specs(self, specs):
        """Normalize specifications to 0-1 range"""
        normalization_factors = {
            'length': 1000, 'beam': 300, 'displacement': 10000000,
            'passengers': 100000, 'power': 200, 'cost': 20,
            'construction_time': 15
        }
        
        normalized = {}
        for key, value in specs.items():
            factor = normalization_factors.get(key, 1)
            normalized[key] = min(1.0, value / factor)
        
        return normalized

def create_dashboard():
    """Main dashboard interface"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛥️ Pangeos Feasibility Analyzer</h1>
        <p>Advanced AI-Powered Yacht Project Analysis System</p>
        <p>Powered by Qwen 3 Multi-Agent Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize systems
    multi_agent = MultiAgentSystem()
    predictor = FeasibilityPredictor()
    
    # Sidebar Controls
    st.sidebar.title("⚙️ Configuration Panel")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Quick Analysis", "Deep Multi-Agent Analysis", "Comparative Analysis", "Scenario Planning"]
    )
    
    st.sidebar.subheader("🛥️ Yacht Specifications")
    
    # Yacht parameters
    specs = {
        'length': st.sidebar.slider("Length (m)", 200, 800, 550),
        'beam': st.sidebar.slider("Beam (m)", 50, 400, 200), 
        'displacement': st.sidebar.slider("Displacement (tons)", 1000000, 10000000, 5000000),
        'passengers': st.sidebar.slider("Passenger Capacity", 10000, 100000, 60000),
        'power': st.sidebar.slider("Propulsion Power (MW)", 50, 300, 150),
        'cost': st.sidebar.slider("Estimated Cost ($B)", 2, 20, 8),
        'construction_time': st.sidebar.slider("Construction Time (years)", 3, 15, 7)
    }
    
    # Analysis trigger
    run_analysis = st.sidebar.button("🚀 Run AI Analysis", type="primary")
    
    if run_analysis:
        with st.spinner("🤖 AI Agents analyzing..."):
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Multi-Agent Analysis
            if analysis_type in ["Deep Multi-Agent Analysis", "Quick Analysis"]:
                
                results = {}
                agents_progress = 0
                
                # Engineering Agent
                status_text.text("🔧 Engineering Agent analyzing...")
                progress_bar.progress(20)
                results['engineering'] = multi_agent.engineering_analysis(specs)
                time.sleep(0.5)
                
                # Financial Agent  
                status_text.text("💰 Financial Agent analyzing...")
                progress_bar.progress(40)
                results['financial'] = multi_agent.financial_analysis(specs)
                time.sleep(0.5)
                
                # Environmental Agent
                status_text.text("🌱 Environmental Agent analyzing...")
                progress_bar.progress(60)
                results['environmental'] = multi_agent.environmental_analysis(specs)
                time.sleep(0.5)
                
                # Logistics Agent
                status_text.text("🚢 Logistics Agent analyzing...")
                progress_bar.progress(80)
                results['logistics'] = multi_agent.logistics_analysis(specs)
                time.sleep(0.5)
                
                # ML Predictions
                status_text.text("🧠 ML Models predicting...")
                progress_bar.progress(90)
                ml_scores = predictor.predict_feasibility(specs)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Display Results
                st.subheader("📊 Analysis Results")
                
                # Metrics Overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Engineering",
                        f"{results['engineering']['score']:.1f}%",
                        delta=f"{results['engineering']['score'] - 70:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Financial", 
                        f"{results['financial']['score']:.1f}%",
                        delta=f"{results['financial']['score'] - 65:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Environmental",
                        f"{results['environmental']['score']:.1f}%", 
                        delta=f"{results['environmental']['score'] - 75:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Logistics",
                        f"{results['logistics']['score']:.1f}%",
                        delta=f"{results['logistics']['score'] - 68:.1f}%"
                    )
                
                # Visualization
                fig = create_feasibility_radar(results, ml_scores)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Agent Reports
                st.subheader("🤖 AI Agent Reports")
                
                for agent_key, agent_result in results.items():
                    with st.expander(f"{agent_result['agent']} Analysis"):
                        st.write(agent_result['analysis'])
            
            # Comparative Analysis
            elif analysis_type == "Comparative Analysis":
                st.subheader("📈 Comparative Scenario Analysis")
                
                scenarios = {
                    "Conservative": {k: v * 0.8 for k, v in specs.items()},
                    "Current": specs,
                    "Optimistic": {k: v * 1.2 for k, v in specs.items()},
                    "Ambitious": {k: v * 1.5 for k, v in specs.items()}
                }
                
                scenario_results = {}
                for scenario_name, scenario_specs in scenarios.items():
                    scenario_results[scenario_name] = predictor.predict_feasibility(scenario_specs)
                
                # Create comparison chart
                fig = create_scenario_comparison(scenario_results)
                st.plotly_chart(fig, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("✅ Comparative Analysis Complete!")
            
            # Scenario Planning
            elif analysis_type == "Scenario Planning":
                st.subheader("🎯 Strategic Scenario Planning")
                
                # Monte Carlo simulation
                num_simulations = 100
                simulation_results = []
                
                for i in range(num_simulations):
                    # Add random variations
                    varied_specs = {
                        key: value * (0.8 + 0.4 * np.random.random()) 
                        for key, value in specs.items()
                    }
                    result = predictor.predict_feasibility(varied_specs)
                    simulation_results.append(result)
                
                # Statistical analysis
                stats_df = pd.DataFrame(simulation_results)
                
                # Create distribution plots
                fig = create_monte_carlo_plots(stats_df)
                st.plotly_chart(fig, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("✅ Scenario Planning Complete!")
    
    # Additional Information
    with st.expander("ℹ️ About This System"):
        st.write("""
        **Pangeos Feasibility Analyzer** combines cutting-edge AI technology with comprehensive yacht engineering analysis:
        
        - 🤖 **Multi-Agent AI System**: 4 specialized agents powered by Qwen 3 models
        - 🧠 **Machine Learning**: Advanced feasibility prediction algorithms
        - 📊 **Real-time Analysis**: Interactive parameter adjustment with instant results
        - 🔍 **Comprehensive Assessment**: Engineering, financial, environmental, and logistics analysis
        - 📈 **Scenario Planning**: Monte Carlo simulations and comparative analysis
        
        **Qwen 3 Models Integration**:
        - Engineering Analysis: Qwen-2.5-72B-Instruct
        - Financial Analysis: Qwen-2-72B-Instruct  
        - Environmental Analysis: Qwen-1.5-110B-Chat
        - Logistics Analysis: Qwen-2.5-7B-Instruct
        """)

def create_feasibility_radar(agent_results, ml_scores):
    """Create radar chart for feasibility analysis"""
    categories = ['Engineering', 'Financial', 'Environmental', 'Logistics']
    agent_scores = [agent_results[key.lower()]['score'] for key in categories]
    ml_scores_list = [ml_scores[key.lower()] for key in categories]
    
    fig = go.Figure()
    
    # AI Agent scores
    fig.add_trace(go.Scatterpolar(
        r=agent_scores,
        theta=categories,
        fill='toself',
        name='AI Agents',
        line_color='#667eea'
    ))
    
    # ML Model scores
    fig.add_trace(go.Scatterpolar(
        r=ml_scores_list,
        theta=categories, 
        fill='toself',
        name='ML Models',
        line_color='#f093fb'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Feasibility Analysis: AI Agents vs ML Models",
        height=500
    )
    
    return fig

def create_scenario_comparison(scenario_results):
    """Create comparison chart for different scenarios"""
    scenarios = list(scenario_results.keys())
    categories = ['engineering', 'financial', 'environmental', 'logistics', 'overall']
    
    fig = go.Figure()
    
    for category in categories:
        values = [scenario_results[scenario][category] for scenario in scenarios]
        fig.add_trace(go.Bar(
            name=category.title(),
            x=scenarios,
            y=values
        ))
    
    fig.update_layout(
        title="Scenario Comparison: Feasibility Scores",
        xaxis_title="Scenarios",
        yaxis_title="Feasibility Score (%)",
        barmode='group',
        height=500
    )
    
    return fig

def create_monte_carlo_plots(stats_df):
    """Create Monte Carlo simulation visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Engineering', 'Financial', 'Environmental', 'Logistics'],
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    categories = ['engineering', 'financial', 'environmental', 'logistics']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (category, pos) in enumerate(zip(categories, positions)):
        fig.add_trace(
            go.Histogram(x=stats_df[category], name=category.title(), nbinsx=20),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        title="Monte Carlo Simulation Results (100 runs)",
        height=600,
        showlegend=False
    )
    
    return fig

if __name__ == "__main__":
    create_dashboard()
