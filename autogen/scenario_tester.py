"""
AutoGen Multi-Scenario Tester
=============================

This script demonstrates AutoGen's flexibility by running different scenarios:
1. Plan a 3-day conference agenda
2. Design a marketing strategy for a product
3. Create a research paper outline
4. Plan a software architecture

Each scenario uses a 4-agent workflow adapted to the specific task.
"""

from datetime import datetime
from config import Config
import json
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI client is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class ScenarioWorkflow:
    """Flexible workflow handler for multiple scenarios"""

    def __init__(self, scenario_type):
        """Initialize the workflow for a specific scenario"""
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE)
        self.outputs = {}
        self.model = Config.OPENAI_MODEL
        self.scenario_type = scenario_type
        
        # Define scenario-specific configurations
        self.scenarios = {
            "conference": {
                "title": "3-Day AI/ML Conference Planning",
                "agents": {
                    "research": "Conference Research Specialist",
                    "analysis": "Attendee & Speaker Analyst",
                    "blueprint": "Agenda Designer",
                    "review": "Conference Director"
                },
                "context": "planning a 3-day AI/ML conference for 500 attendees"
            },
            "marketing": {
                "title": "Product Marketing Strategy",
                "agents": {
                    "research": "Market Research Analyst",
                    "analysis": "Customer Insights Specialist",
                    "blueprint": "Marketing Strategist",
                    "review": "CMO Reviewer"
                },
                "context": "creating a marketing strategy for a new B2B SaaS product"
            },
            "research_paper": {
                "title": "Research Paper Outline",
                "agents": {
                    "research": "Literature Review Specialist",
                    "analysis": "Research Gap Analyst",
                    "blueprint": "Paper Structure Designer",
                    "review": "Academic Editor"
                },
                "context": "outlining a research paper on AI ethics in healthcare"
            },
            "software": {
                "title": "Software Architecture Planning",
                "agents": {
                    "research": "Tech Stack Researcher",
                    "analysis": "Requirements Analyst",
                    "blueprint": "System Architect",
                    "review": "CTO Reviewer"
                },
                "context": "designing architecture for a scalable e-commerce platform"
            }
        }

    def run(self):
        """Execute the complete workflow"""
        scenario = self.scenarios[self.scenario_type]
        
        print("\n" + "="*80)
        print(f"AUTOGEN SCENARIO TEST: {scenario['title']}")
        print("="*80)
        print(f"Scenario: {scenario['context']}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model}\n")

        # Execute phases based on scenario type
        self.phase_research(scenario)
        self.phase_analysis(scenario)
        self.phase_blueprint(scenario)
        self.phase_review(scenario)
        self.print_summary(scenario)

    def phase_research(self, scenario):
        """Phase 1: Research"""
        print("\n" + "="*80)
        print("PHASE 1: RESEARCH")
        print("="*80)
        print(f"[{scenario['agents']['research']} is working...]")

        prompts = {
            "conference": """You are a conference research specialist. Research successful AI/ML conferences 
and provide insights on: keynote speakers, workshop formats, networking events, and sponsorship models.
Analyze 3 similar conferences. Be concise - 150 words.""",

            "marketing": """You are a market research analyst. Research the B2B SaaS market and provide 
insights on: target personas, competitor strategies, content marketing trends, and channel effectiveness.
Focus on data-driven insights. Be concise - 150 words.""",

            "research_paper": """You are a literature review specialist. Research existing papers on AI ethics 
in healthcare and provide: key themes, major contributors, methodologies used, and recent developments.
Identify 3-4 seminal papers. Be concise - 150 words.""",

            "software": """You are a tech stack researcher. Research modern e-commerce architectures and 
provide insights on: microservices patterns, database choices, API designs, and scalability solutions.
Compare 3 architecture approaches. Be concise - 150 words."""
        }

        user_messages = {
            "conference": "Research successful AI/ML conferences and identify best practices.",
            "marketing": "Research the B2B SaaS market and identify effective marketing strategies.",
            "research_paper": "Research existing literature on AI ethics in healthcare.",
            "software": "Research modern e-commerce platform architectures and technologies."
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts[self.scenario_type]},
                {"role": "user", "content": user_messages[self.scenario_type]}
            ]
        )

        self.outputs["research"] = response.choices[0].message.content
        print(f"\n[{scenario['agents']['research']} Output]")
        print(self.outputs["research"])

    def phase_analysis(self, scenario):
        """Phase 2: Analysis"""
        print("\n" + "="*80)
        print("PHASE 2: ANALYSIS")
        print("="*80)
        print(f"[{scenario['agents']['analysis']} is working...]")

        prompts = {
            "conference": """You are an attendee and speaker analyst. Based on the research, identify:
target attendee profiles, speaker selection criteria, and engagement opportunities.
Provide 3 key insights. Be concise - 150 words.""",

            "marketing": """You are a customer insights specialist. Based on the research, identify:
customer pain points, buying journey stages, and value proposition opportunities.
Provide 3 actionable insights. Be concise - 150 words.""",

            "research_paper": """You are a research gap analyst. Based on the literature review, identify:
unexplored areas, methodological improvements, and potential contributions.
Highlight 3 research gaps. Be concise - 150 words.""",

            "software": """You are a requirements analyst. Based on the research, identify:
critical system requirements, performance needs, and integration challenges.
List 3 key requirements. Be concise - 150 words."""
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts[self.scenario_type]},
                {"role": "user", "content": f"Research findings:\n{self.outputs['research']}\n\nProvide your analysis."}
            ]
        )

        self.outputs["analysis"] = response.choices[0].message.content
        print(f"\n[{scenario['agents']['analysis']} Output]")
        print(self.outputs["analysis"])

    def phase_blueprint(self, scenario):
        """Phase 3: Blueprint/Design"""
        print("\n" + "="*80)
        print("PHASE 3: BLUEPRINT/DESIGN")
        print("="*80)
        print(f"[{scenario['agents']['blueprint']} is working...]")

        prompts = {
            "conference": """You are an agenda designer. Create a detailed 3-day conference agenda including:
Day-by-day schedule with timings, keynote topics, workshop themes, and networking events.
Be specific and practical - 150 words.""",

            "marketing": """You are a marketing strategist. Design a comprehensive strategy including:
Campaign objectives, key messages, channel mix, content calendar outline, and success metrics.
Be specific and actionable - 150 words.""",

            "research_paper": """You are a paper structure designer. Create a detailed outline including:
Section titles, key arguments per section, methodology approach, and expected contributions.
Follow academic standards - 150 words.""",

            "software": """You are a system architect. Design the architecture including:
System components, data flow, API structure, deployment strategy, and scalability approach.
Include specific technologies - 150 words."""
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts[self.scenario_type]},
                {"role": "user", "content": f"Analysis:\n{self.outputs['analysis']}\n\nCreate your design/blueprint."}
            ]
        )

        self.outputs["blueprint"] = response.choices[0].message.content
        print(f"\n[{scenario['agents']['blueprint']} Output]")
        print(self.outputs["blueprint"])

    def phase_review(self, scenario):
        """Phase 4: Review"""
        print("\n" + "="*80)
        print("PHASE 4: REVIEW & RECOMMENDATIONS")
        print("="*80)
        print(f"[{scenario['agents']['review']} is working...]")

        prompts = {
            "conference": """You are a conference director. Review the agenda and provide:
Risk mitigation strategies, budget considerations, and success metrics.
Give 3 strategic recommendations - 150 words.""",

            "marketing": """You are a CMO reviewer. Review the strategy and provide:
Budget allocation advice, risk factors, and optimization opportunities.
Give 3 strategic improvements - 150 words.""",

            "research_paper": """You are an academic editor. Review the outline and provide:
Strengths, potential weaknesses, and publication strategy recommendations.
Give 3 improvement suggestions - 150 words.""",

            "software": """You are a CTO reviewer. Review the architecture and provide:
Security considerations, cost optimization, and deployment recommendations.
Give 3 technical improvements - 150 words."""
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts[self.scenario_type]},
                {"role": "user", "content": f"Blueprint:\n{self.outputs['blueprint']}\n\nProvide your review."}
            ]
        )

        self.outputs["review"] = response.choices[0].message.content
        print(f"\n[{scenario['agents']['review']} Output]")
        print(self.outputs["review"])

    def print_summary(self, scenario):
        """Print final summary"""
        print("\n" + "="*80)
        print("WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"""
Scenario: {scenario['title']}
Context: {scenario['context']}

Agents Used:
1. {scenario['agents']['research']} - Gathered initial research
2. {scenario['agents']['analysis']} - Analyzed findings
3. {scenario['agents']['blueprint']} - Created the design/plan
4. {scenario['agents']['review']} - Provided strategic review

Each agent built upon the previous agent's output, demonstrating
AutoGen's sequential workflow capabilities.
""")

        # Save outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"scenario_{self.scenario_type}_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"AUTOGEN SCENARIO: {scenario['title']}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model}\n\n")
            
            for phase in ["research", "analysis", "blueprint", "review"]:
                f.write("\n" + "-"*80 + "\n")
                f.write(f"{phase.upper()} PHASE\n")
                f.write("-"*80 + "\n")
                f.write(self.outputs[phase] + "\n")
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)


def main():
    """Main function to run scenarios"""
    print("\n" + "="*80)
    print("AUTOGEN MULTI-SCENARIO TESTER")
    print("="*80)
    print("\nAvailable scenarios:")
    print("1. conference - Plan a 3-day AI/ML conference")
    print("2. marketing - Design a B2B SaaS marketing strategy")
    print("3. research_paper - Create an AI ethics research paper outline")
    print("4. software - Plan an e-commerce platform architecture")
    
    if len(sys.argv) > 1:
        scenario_type = sys.argv[1]
    else:
        scenario_type = input("\nEnter scenario (conference/marketing/research_paper/software): ").strip()
    
    valid_scenarios = ["conference", "marketing", "research_paper", "software"]
    if scenario_type not in valid_scenarios:
        print(f"\n❌ Invalid scenario. Please choose from: {', '.join(valid_scenarios)}")
        exit(1)
    
    try:
        workflow = ScenarioWorkflow(scenario_type)
        workflow.run()
        print("\n✅ Scenario completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()