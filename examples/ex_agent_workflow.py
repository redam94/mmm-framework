"""
Example: Agentic MMM Workflow
=============================

This example demonstrates how to use the new agentic framework to
build, configure, and fit a Marketing Mix Model interactively using LangGraph.

The LLM is selected by the model configuration file rather than hard-coded
here. Either copy `config/model_config.example.yaml` to
`config/model_config.yaml` and edit it, or set an API key env var for a direct
provider (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY). For Vertex AI on
a GCP VM, set `provider: vertex_anthropic` (or `vertex_gemini`) in the config
file -- Application Default Credentials are used automatically.

Usage:
    # Direct provider (no config file needed):
    export ANTHROPIC_API_KEY="your-api-key"
    python examples/ex_agent_workflow.py

    # Or Vertex AI / ADC on a GCP VM:
    cp config/model_config.example.yaml config/model_config.yaml  # then edit
    python examples/ex_agent_workflow.py
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.markdown import Markdown

from mmm_framework.agents import build_llm, create_agent_graph, load_model_config

def main():
    # 1. Initialize the LLM from the model configuration file (+ env overrides).
    #    See config/model_config.example.yaml for all options, including Vertex
    #    AI providers that authenticate via Application Default Credentials.
    print("Initializing LLM and Agent Graph...")

    cfg = load_model_config()
    if cfg.uses_adc:
        auth = "Application Default Credentials"
    elif cfg.uses_vertex:
        auth = "explicit service-account key"
    else:
        auth = "API key"
    print(f"Provider: {cfg.provider} | Model: {cfg.model} | Auth: {auth}")

    try:
        llm = build_llm(cfg)
    except Exception as exc:
        raise SystemExit(
            f"Failed to initialise the LLM ({cfg.provider}): {exc}\n"
            "Set an API key env var (ANTHROPIC_API_KEY / OPENAI_API_KEY / "
            "GOOGLE_API_KEY) for a direct provider, or configure Vertex AI in "
            "config/model_config.yaml."
        ) from exc

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
