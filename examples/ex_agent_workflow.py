"""
Example: Agentic MMM Workflow
=============================

This example demonstrates how to use the new agentic framework to 
build, configure, and fit a Marketing Mix Model interactively using LangGraph.

Usage:
    export GOOGLE_API_KEY="your-api-key"
    python examples/ex_agent_workflow.py
"""

import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.markdown import Markdown

from mmm_framework.agents import create_agent_graph

def main():
    # 1. Initialize the LLM
    # Make sure you have GOOGLE_API_KEY set in your environment
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to run this example: export GOOGLE_API_KEY='your-api-key'")
        # For the sake of the example, we will let it fail later if not set,
        # but you could mock the LLM here for testing.
        
    print("Initializing LLM and Agent Graph...")
    
    # You can easily swap out the LLM provider here.
    # Make sure to set the appropriate environment variable (GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY)
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        print("Using Anthropic Claude 3.5 Sonnet")
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    elif os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        print("Using OpenAI GPT-4o")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Using Google Gemini 1.5 Pro")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    else:
        raise ValueError("No API key found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY.")
    
    
    # Compile the graph with a memory checkpointer to keep conversation state
    memory = MemorySaver()
    agent_graph = create_agent_graph(llm, checkpointer=memory)
    
    # 2. Define the thread configuration for memory
    config = {"configurable": {"thread_id": "mmm_demo_thread_1"}}
    
    console = Console()
    
    # 3. Simulate a user conversation
    console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
    console.print("[bold green]User:[/bold green] I want to build a Marketing Mix Model, but I don't have data. Can you generate some and run a model?")
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    
    # Keep track of outputs for the markdown file
    markdown_output = [f"# Agentic MMM Workflow\n"]
    
    # Keep track of printed message IDs so we don't print them twice
    printed_messages = set()
    
    user_input = "I want to build a Marketing Mix Model, but I don't have data. Can you generate some and run a model?"
    
    while True:
        console.print(f"[bold green]User:[/bold green] {user_input}")
        markdown_output.append(f"**User:** {user_input}\n")
        
        initial_message = HumanMessage(content=user_input)
        
        console.print("\n[italic yellow]Agent is thinking and executing tools...[/italic yellow]\n")
        
        # We stream the events to see what the agent is doing
        for event in agent_graph.stream({"messages": [initial_message]}, config, stream_mode="values"):
            # The stream emits the full state at each step.
            latest_message = event["messages"][-1]
            
            if not hasattr(latest_message, "id") or not latest_message.id:
                # If there's no ID, use the object id or content hash as fallback
                msg_id = hash(str(latest_message))
            else:
                msg_id = latest_message.id
                
            if msg_id in printed_messages:
                continue
                
            printed_messages.add(msg_id)
            
            # If it's an AI message, print its content
            if hasattr(latest_message, "content") and latest_message.content:
                if latest_message.type == "ai":
                    # Some Anthropic responses have a list of dicts for content
                    content = latest_message.content
                    if isinstance(content, list):
                        text_content = next((item['text'] for item in content if isinstance(item, dict) and 'text' in item), str(content))
                    else:
                        text_content = str(content)
                        
                    console.print(Markdown(f"**Agent:**\n{text_content}"))
                    markdown_output.append(f"**Agent:**\n{text_content}\n")
            
            # If it's a tool call, print what it's calling
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    console.print(f"[bold magenta][Tool Call][/bold magenta] {tc['name']}({tc['args']})")
                    markdown_output.append(f"> **Tool Call:** `{tc['name']}({tc['args']})`\n")
                    
            # If it's a tool output, print it
            if latest_message.type == "tool":
                console.print(f"[bold blue][Tool Result][/bold blue] {latest_message.name} completed.\n")
                markdown_output.append(f"> **Tool Result:** `{latest_message.name}` completed.\n")

        # Ask user for next input
        console.print("\n[bold cyan]" + "-"*60 + "[/bold cyan]")
        user_input = console.input("[bold green]You (type 'exit' to stop):[/bold green] ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        console.print()

    console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
    console.print("[bold green]Workflow Complete![/bold green]")
    
    # Retrieve the final state to show the result
    final_state = agent_graph.get_state(config).values
    console.print("\n[bold]Final State Summary:[/bold]")
    console.print(f"Dataset: [green]{final_state.get('dataset_path')}[/green]")
    console.print(f"Model Status: [green]{final_state.get('model_status')}[/green]")
    if final_state.get('report_path'):
        console.print(f"Report Location: [green]{final_state.get('report_path')}[/green]")
        
    # Write to markdown file
    output_filename = "agent_workflow_output.md"
    with open(output_filename, "w") as f:
        f.write("\n".join(markdown_output))
    console.print(f"\n[italic]A complete formatted transcript was saved to {output_filename}[/italic]")

if __name__ == "__main__":
    main()
