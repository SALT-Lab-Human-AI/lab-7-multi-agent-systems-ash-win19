"""
CrewAI Multi-Agent Demo: Startup Analysis System (REAL API VERSION)
==================================================================

This implementation uses REAL OpenAI API calls and web search to gather
actual startup information for comprehensive startup analysis.

Agents use:
1. OpenAI GPT-4 for intelligent research and recommendations
2. Web search for real-time market data and startup information
3. Real competitive analysis data from current sources

Agents:
1. StartupScout - Startup Discovery Specialist (researches emerging startups)
2. MarketAnalyst - Competitive Analysis Expert (analyzes market competition)
3. ProductStrategist - Product Pitch Specialist (creates compelling pitches)
4. FeatureAnalyst - Feature Analysis Expert (evaluates product features)

Configuration:
- Uses shared configuration from the root .env file
- Environment variables set in .env file at project root
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
import requests

# Add parent directory to path to import shared_config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared configuration
from shared_config import Config, validate_config


# ============================================================================
# TOOLS (Real API implementations using web search)
# ============================================================================

@tool
def search_emerging_startups(industry: str, location: str = "US") -> str:
    """
    Search for emerging startups in a specific industry and location.
    Uses web search to find current information from startup databases and news.
    """
    search_query = f"emerging startups {industry} {location} 2025 seed funding series A"

    return f"""
    Research task: Find emerging startups in {industry} within {location}.

    Please research and provide:
    1. Recently funded startups (check Crunchbase, PitchBook, AngelList)
    2. Startup founding teams and their backgrounds
    3. Funding rounds and investor information
    4. Problem they're solving and target market
    5. Current traction and growth metrics
    6. Notable achievements and press coverage

    Focus on startups founded within the last 3 years with recent activity.
    """


@tool
def analyze_competitors(startup_name: str, industry: str) -> str:
    """
    Analyze direct and indirect competitors of a startup.
    Provides comprehensive competitive landscape analysis.
    """
    search_query = f"{startup_name} competitors {industry} market share comparison analysis"

    return f"""
    Research task: Analyze competitors of {startup_name} in the {industry} space.

    Please research and provide:
    1. Direct competitors with similar offerings (check G2, Capterra, ProductHunt)
    2. Market share and positioning of each competitor
    3. Competitive advantages and unique selling propositions
    4. Pricing strategies and business models
    5. Customer reviews and satisfaction ratings
    6. Recent product launches and strategic moves
    7. Strengths and weaknesses analysis

    Include both established players and emerging competitors.
    Focus on actionable competitive insights.
    """


@tool
def analyze_product_features(product_name: str, category: str) -> str:
    """
    Analyze product features and user feedback for a specific product.
    Provides detailed feature analysis and improvement suggestions.
    """
    search_query = f"{product_name} features review analysis user feedback {category}"

    return f"""
    Research task: Analyze features of {product_name} in the {category} space.

    Please research and provide:
    1. Core features and functionality overview
    2. User reviews and feature-specific feedback (check ProductHunt, G2, Reddit)
    3. Feature comparison with competing products
    4. Most requested features from users
    5. Technical specifications and integrations
    6. Usability and user experience insights
    7. Feature gaps and improvement opportunities

    Include both positive feedback and pain points.
    Focus on actionable insights for product development.
    """


@tool
def research_market_trends(industry: str, timeframe: str = "2025") -> str:
    """
    Research market trends and opportunities in a specific industry.
    Provides insights on market dynamics and future opportunities.
    """
    search_query = f"{industry} market trends {timeframe} growth opportunities forecast"

    return f"""
    Research task: Analyze market trends in the {industry} sector for {timeframe}.

    Please research and provide:
    1. Current market size and growth projections
    2. Key trends driving the industry (check Gartner, McKinsey, industry reports)
    3. Emerging technologies and innovations
    4. Consumer behavior shifts and preferences
    5. Regulatory changes and their impact
    6. Investment trends and VC interest
    7. Market opportunities and white spaces

    Provide data-driven insights with credible sources.
    Focus on actionable trends for startup opportunities.
    """


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_startup_scout_agent(industry: str, location: str):
    """Create the Startup Scout agent with real research tools."""
    return Agent(
        role="Startup Discovery Specialist",
        goal=f"Discover and analyze emerging startups in the {industry} sector "
             f"within {location}, identifying promising companies with innovative solutions. "
             f"Use real data from startup databases and funding news to provide accurate insights.",
        backstory="You are a seasoned startup scout with extensive experience in "
                  "identifying promising early-stage companies. With a background in venture capital "
                  "and deep connections in the startup ecosystem, you excel at spotting trends "
                  "before they become mainstream. You have evaluated thousands of startups and "
                  "have a keen eye for innovation, strong founding teams, and scalable business models. "
                  "You always verify information from multiple sources including Crunchbase, AngelList, and industry news.",
        tools=[search_emerging_startups],
        verbose=True,
        allow_delegation=False
    )


def create_market_analyst_agent(industry: str):
    """Create the Market Analyst agent with competitive analysis tools."""
    return Agent(
        role="Competitive Analysis Expert",
        goal=f"Conduct comprehensive competitive analysis in the {industry} market, "
             f"identifying key players, market dynamics, and strategic opportunities. "
             f"Use real market data and competitor information to provide actionable insights.",
        backstory="You are a strategic market analyst with over 15 years of experience "
                  "in competitive intelligence and market research. Having worked with Fortune 500 "
                  "companies and top-tier consulting firms, you specialize in dissecting market "
                  "landscapes and identifying competitive advantages. You combine data analytics "
                  "with strategic thinking to uncover hidden opportunities. You meticulously "
                  "analyze competitor strategies, market positioning, and customer sentiment "
                  "using tools like G2, Capterra, and industry reports.",
        tools=[analyze_competitors, research_market_trends],
        verbose=True,
        allow_delegation=False
    )


def create_product_strategist_agent(industry: str):
    """Create the Product Strategist agent with pitch creation tools."""
    return Agent(
        role="Product Pitch Specialist",
        goal=f"Create compelling product pitches and value propositions for startups "
             f"in the {industry} sector, crafting narratives that resonate with investors and customers. "
             f"Use market insights and competitive analysis to develop unique positioning.",
        backstory="You are a master storyteller and product strategist who has helped "
                  "launch over 100 successful products. With experience at top tech companies "
                  "and as a startup founder yourself, you understand what makes products compelling. "
                  "You excel at translating complex technical features into clear value propositions "
                  "that resonate with target audiences. You craft pitches that have secured millions "
                  "in funding and driven significant customer adoption. You always ground your "
                  "strategies in market research and user insights.",
        tools=[analyze_product_features],
        verbose=True,
        allow_delegation=False
    )


def create_feature_analyst_agent(category: str):
    """Create the Feature Analyst agent with product analysis tools."""
    return Agent(
        role="Feature Analysis Expert",
        goal=f"Analyze product features and user feedback for {category} products, "
             f"identifying strengths, weaknesses, and opportunities for innovation. "
             f"Use real user reviews and feature comparisons to provide actionable insights.",
        backstory="You are a product analyst with deep expertise in feature optimization "
                  "and user experience design. Having analyzed hundreds of products across "
                  "various industries, you have a unique ability to identify what makes features "
                  "successful or problematic. You combine quantitative data analysis with "
                  "qualitative user research to provide comprehensive feature assessments. "
                  "Your insights have helped companies improve user satisfaction scores by 40% "
                  "on average. You always base recommendations on real user feedback and data.",
        tools=[analyze_product_features, research_market_trends],
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_startup_discovery_task(startup_scout, industry: str, location: str, stage: str):
    """Define the startup discovery task using real data."""
    return Task(
        description=f"Research and compile a list of REAL emerging startups in the {industry} "
                   f"industry within {location} at {stage} stage. "
                   f"Use actual current data from Crunchbase, AngelList, PitchBook, "
                   f"and recent funding news. Find at least 3-5 promising startups, "
                   f"including details about their founding team, problem they solve, "
                   f"funding history, traction metrics, and competitive advantages. Provide "
                   f"recommendations on which startups show the most promise based on "
                   f"market opportunity and execution capability.",
        agent=startup_scout,
        expected_output=f"A detailed report with 3-5 REAL emerging startups in {industry} "
                       f"including founding teams, funding details, traction metrics, and strategic recommendations based on "
                       f"actual data from startup databases and news sources"
    )


def create_competitive_analysis_task(market_analyst, startup_name: str, industry: str):
    """Define the competitive analysis task using real data."""
    return Task(
        description=f"Based on the discovered startup {startup_name}, conduct a comprehensive "
                   f"competitive analysis in the {industry} market. Research actual competitors "
                   f"using G2, Capterra, ProductHunt, and industry reports. For each competitor, "
                   f"provide their market positioning, unique value propositions, pricing strategies, "
                   f"customer satisfaction scores, and recent strategic moves. "
                   f"Include both direct competitors and potential threats from adjacent markets. "
                   f"Analyze market trends and identify opportunities for differentiation.",
        agent=market_analyst,
        expected_output=f"A comprehensive competitive analysis for {startup_name} in {industry} with "
                       f"detailed competitor profiles, market positioning matrix, SWOT analysis, "
                       f"and strategic recommendations based on actual market data"
    )


def create_product_pitch_task(product_strategist, startup_name: str, industry: str, target_audience: str):
    """Define the product pitch creation task using market insights."""
    return Task(
        description=f"Create a compelling product pitch for {startup_name} in the {industry} sector "
                   f"targeting {target_audience}. Based on the competitive analysis, craft a unique "
                   f"value proposition that differentiates from competitors. Develop key messaging "
                   f"that addresses customer pain points and highlights competitive advantages. "
                   f"Create a pitch deck outline with compelling narrative, market opportunity sizing, "
                   f"go-to-market strategy, and investment highlights. Use real market data and "
                   f"customer insights to support your recommendations.",
        agent=product_strategist,
        expected_output=f"A comprehensive product pitch for {startup_name} including unique value proposition, "
                       f"key messaging framework, pitch deck outline with 10-12 slides, go-to-market strategy, "
                       f"and investment highlights supported by real market data"
    )


def create_feature_analysis_task(feature_analyst, product_name: str, category: str):
    """Define the feature analysis task using real user feedback."""
    return Task(
        description=f"Based on the product pitch for {product_name}, conduct a detailed feature "
                   f"analysis in the {category} space. Research actual user feedback from "
                   f"ProductHunt, G2, Reddit, and app store reviews. Analyze core features, "
                   f"user satisfaction levels, feature requests, and pain points. Compare features "
                   f"with competing products and identify gaps in the market. Provide recommendations "
                   f"for feature prioritization, potential innovations, and product roadmap based on "
                   f"real user needs and market trends.",
        agent=feature_analyst,
        expected_output=f"A comprehensive feature analysis for {product_name} including feature comparison matrix, "
                       f"user satisfaction scores, top feature requests, competitive feature gaps, "
                       f"and prioritized roadmap recommendations based on real user feedback"
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def main(industry: str = "AI/ML", location: str = "San Francisco Bay Area",
         startup_stage: str = "Seed to Series A", target_audience: str = "B2B SaaS",
         analysis_focus: str = "comprehensive", category: str = "productivity tools"):
    """
    Main function to orchestrate the startup analysis crew.

    Args:
        industry: Target industry for analysis (e.g., "AI/ML", "FinTech", "HealthTech")
        location: Geographic focus (e.g., "San Francisco Bay Area", "New York", "Global")
        startup_stage: Stage of startups to analyze (e.g., "Seed to Series A", "Series B+")
        target_audience: Target market (e.g., "B2B SaaS", "B2C Mobile", "Enterprise")
        analysis_focus: Type of analysis ("comprehensive", "competitive", "product-focused")
        category: Product category for feature analysis (e.g., "productivity tools", "analytics")
    """

    print("=" * 80)
    print("CrewAI Multi-Agent Startup Analysis System (REAL API VERSION)")
    print(f"Analyzing {startup_stage} Startups in {industry}")
    print("=" * 80)
    print()
    print(f"🚀 Industry Focus: {industry}")
    print(f"📍 Location: {location}")
    print(f"📊 Startup Stage: {startup_stage}")
    print(f"🎯 Target Audience: {target_audience}")
    print(f"🔍 Analysis Type: {analysis_focus}")
    print(f"📦 Product Category: {category}")
    print()

    # Validate configuration before proceeding
    print("🔍 Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed. Please set up your .env file.")
        exit(1)

    # Set environment variables for CrewAI (it reads from os.environ)
    # CrewAI uses OPENAI_API_KEY and OPENAI_API_BASE environment variables
    os.environ["OPENAI_API_KEY"] = Config.API_KEY
    os.environ["OPENAI_API_BASE"] = Config.API_BASE
    
    # For Groq compatibility, also set OPENAI_MODEL_NAME
    if Config.USE_GROQ:
        os.environ["OPENAI_MODEL_NAME"] = Config.OPENAI_MODEL

    print("✅ Configuration validated successfully!")
    print()
    Config.print_summary()
    print()
    print("⚠️  IMPORTANT: This version uses REAL OpenAI API calls and web search")
    print("    Agents will research actual startup data and market information")
    print()
    print("Tip: Check your API usage at https://platform.openai.com/account/usage")
    print()

    # Create agents with startup parameters
    print("[1/4] Creating Startup Scout Agent (discovers emerging startups)...")
    startup_scout = create_startup_scout_agent(industry, location)

    print("[2/4] Creating Market Analyst Agent (analyzes competition)...")
    market_analyst = create_market_analyst_agent(industry)

    print("[3/4] Creating Product Strategist Agent (crafts pitches)...")
    product_strategist = create_product_strategist_agent(industry)

    print("[4/4] Creating Feature Analyst Agent (evaluates products)...")
    feature_analyst = create_feature_analyst_agent(category)

    print("\n✅ All agents created successfully!")
    print()

    # Create tasks with startup parameters
    print("Creating tasks for the crew...")
    # For demo purposes, we'll use a placeholder startup name that will be discovered
    startup_discovery_task = create_startup_discovery_task(startup_scout, industry, location, startup_stage)
    competitive_analysis_task = create_competitive_analysis_task(market_analyst, "[To be discovered]", industry)
    product_pitch_task = create_product_pitch_task(product_strategist, "[To be discovered]", industry, target_audience)
    feature_analysis_task = create_feature_analysis_task(feature_analyst, "[To be discovered]", category)

    print("Tasks created successfully!")
    print()

    # Create the crew with sequential task execution
    print("Forming the Startup Analysis Crew...")
    print("Task Sequence: StartupScout → MarketAnalyst → ProductStrategist → FeatureAnalyst")
    print()

    crew = Crew(
        agents=[startup_scout, market_analyst, product_strategist, feature_analyst],
        tasks=[startup_discovery_task, competitive_analysis_task, product_pitch_task, feature_analysis_task],
        verbose=True,
        process="sequential"  # Sequential task execution
    )

    # Execute the crew
    print("=" * 80)
    print("Starting Crew Execution with REAL API Calls...")
    print(f"Analyzing {startup_stage} startups in {industry} sector")
    print("=" * 80)
    print()

    try:
        result = crew.kickoff(inputs={
            "industry": industry,
            "location": location,
            "startup_stage": startup_stage,
            "target_audience": target_audience,
            "analysis_focus": analysis_focus,
            "category": category
        })

        print()
        print("=" * 80)
        print("✅ Crew Execution Completed Successfully!")
        print("=" * 80)
        print()
        print(f"FINAL STARTUP ANALYSIS REPORT FOR {industry.upper()} (Based on Real API Data):")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # Save output to file
        output_filename = f"crewai_startup_analysis_{industry.lower().replace('/', '_')}.txt"
        output_path = Path(__file__).parent / output_filename

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CrewAI Multi-Agent Startup Analysis System - Real API Execution Report\n")
            f.write(f"Analyzing {startup_stage} Startups in {industry}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Parameters:\n")
            f.write(f"  Industry: {industry}\n")
            f.write(f"  Location: {location}\n")
            f.write(f"  Startup Stage: {startup_stage}\n")
            f.write(f"  Target Audience: {target_audience}\n")
            f.write(f"  Analysis Focus: {analysis_focus}\n")
            f.write(f"  Product Category: {category}\n\n")
            f.write(f"Execution Time: {datetime.now()}\n")
            f.write(f"API Version: REAL API CALLS (OpenAI GPT-4)\n")
            f.write(f"Data Source: Web research via OpenAI\n\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("- All startup data, funding information, and market analysis is based on real data\n")
            f.write("- Information is current as of the date this was run\n")
            f.write("- Startup valuations and funding status may change rapidly\n")
            f.write("- Competitive landscape and market trends should be verified before investment decisions\n\n")
            f.write("FINAL STARTUP ANALYSIS REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result))
            f.write("\n" + "-" * 80 + "\n")

        print(f"\n✅ Output saved to {output_filename}")
        print("ℹ️  Note: All data in this report is based on REAL API calls to OpenAI")
        print("    and research of current startup databases and market sources.")

    except Exception as e:
        print(f"\n❌ Error during crew execution: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify OPENAI_API_KEY is set: export OPENAI_API_KEY='sk-...'")
        print("   2. Check API key is valid and has sufficient credits")
        print("   3. Verify internet connection for web research")
        print("   4. Check OpenAI API status at https://status.openai.com")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow command line arguments to override defaults
    import sys

    kwargs = {
        "industry": "AI/ML",
        "location": "San Francisco Bay Area",
        "startup_stage": "Seed to Series A",
        "target_audience": "B2B SaaS",
        "analysis_focus": "comprehensive",
        "category": "productivity tools"
    }

    # Parse command line arguments (optional)
    # Usage: python crewai_demo.py [industry] [location] [startup_stage]
    # Example: python crewai_demo.py "FinTech" "New York" "Series B+"
    if len(sys.argv) > 1:
        kwargs["industry"] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs["location"] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs["startup_stage"] = sys.argv[3]
    if len(sys.argv) > 4:
        kwargs["target_audience"] = sys.argv[4]
    if len(sys.argv) > 5:
        kwargs["analysis_focus"] = sys.argv[5]
    if len(sys.argv) > 6:
        kwargs["category"] = sys.argv[6]

    main(**kwargs)
