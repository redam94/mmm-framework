"""Example: Running the MMM Agent on Vertex AI with ADC (GCP VM)
================================================================

This is a minimal, non-interactive smoke test for running the MMM agent against
**Google Vertex AI** using **Application Default Credentials (ADC)** — the setup
you'd use on a GCP Compute Engine VM whose attached service account has the
``roles/aiplatform.user`` role.

No API key is involved: ``google.auth.default()`` resolves the VM's service
account automatically. Locally you can reproduce the same path with::

    gcloud auth application-default login

Setup
-----
1. Copy and edit the model configuration::

       cp config/model_config.example.yaml config/model_config.yaml

   Set, for example::

       provider: vertex_anthropic      # or vertex_gemini
       model:    <your Vertex Model Garden id>
       project:  your-gcp-project      # or leave blank to use the VM project
       location: us-east5              # a region that serves your model

   (You can instead use env vars: ``MMM_LLM_PROVIDER``, ``MMM_LLM_MODEL``,
   ``MMM_LLM_PROJECT``, ``MMM_LLM_LOCATION``.)

2. Run it::

       python examples/ex_vertex_agent.py
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from mmm_framework.agents import build_llm, create_agent_graph, describe_active_config


def main() -> None:
    info = describe_active_config()
    print("Active model configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if not info["uses_vertex"]:
        print(
            "\nNote: the active provider is not a Vertex provider. Set "
            "`provider: vertex_anthropic` (or `vertex_gemini`) in "
            "config/model_config.yaml to exercise the ADC path."
        )

    print("\nBuilding the agent graph (LLM via ADC; no API key needed)...")
    llm = build_llm()
    agent = create_agent_graph(llm, checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "vertex_smoke_test"}}
    prompt = (
        "In one sentence, what is the first step of the canonical causal MMM "
        "workflow you follow?"
    )
    print(f"\nUser: {prompt}\n")

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config)
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):  # some providers return a list of content blocks
        content = next(
            (b["text"] for b in content if isinstance(b, dict) and "text" in b),
            str(content),
        )
    print(f"Agent: {content}")
    print("\nSuccess — Vertex AI + ADC round-trip completed.")


if __name__ == "__main__":
    main()
