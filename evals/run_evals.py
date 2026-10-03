import asyncio
from datetime import timezone, datetime
from openevals.json import create_async_json_match_evaluator
from openevals.llm import create_llm_as_judge
from evals.test_dataset import SCENARIOS
# Import your agent invocation pipeline here
from agents.agent import run_production_briefing_agent 

async def main():
    deterministic_evaluator = create_async_json_match_evaluator(list_aggregator="average")
    

    
    # Define a custom evaluation prompt to guide the judge on your tool outputs
    TOOL_CALL_RUBRIC = """
    You are assessing the correctness of a tool call's argument payload.
    Compare the generated tool arguments (outputs) against the reference expectations (reference_outputs).

    - Pay attention to the intent of the parameters (e.g., search keywords, time windows).
    - Ignore instructional wrapper phrases in the reference like "Must search for terms like", "Should look for", or "Must accurately represent". 
    - For date and time parameters (e.g., ISO 8601 bounds for 'today'): Resolve what 'today' means based on the Input context, and check if the generated output correctly represents that 24-hour date window.
    - If the core search query, resolved temporal constraints, or semantic parameters align with the user's intent, give a score of 1. Otherwise, 0.

    Input context: {inputs}
    Generated output: {outputs}
    Expected reference: {reference_outputs}
    """

    # Instantiate the judge
    semantic_json_evaluator = create_llm_as_judge(
        prompt=TOOL_CALL_RUBRIC,
        model="openai:o3-mini",
        feedback_key="semantic_json_correctness"
    )

    print("Starting Sequential Agent Parameter Evaluation....\n")

    for scenario in SCENARIOS:
        print(f"Testing Case: {scenario['id']}")
        print(f"Input: '{scenario['input']}'")
        
        # Execute the agent loop
        agent_output = await run_production_briefing_agent(scenario["input"])
        actual_tool_calls = agent_output.get("tool_calls", [])

        # Calculate how many tool calls are actually expected from the dataset
        expected_tool_calls_count = 0 if (len(scenario["steps"]) == 1 and scenario["steps"][0]["expected_tool"] is None) else len(scenario["steps"])

        if expected_tool_calls_count == 0:
            scenario_passed = True
            print("No tool calls expected for this scenario.")            
        else:
            scenario_passed = False

        if expected_tool_calls_count > 0:
            # Iterate through steps sequentially
            for idx, expected_step in enumerate(scenario["steps"]):
                actual_call = actual_tool_calls[idx]
                actual_tool = actual_call["name"]
                actual_args = actual_call["parameters"]

                print(f"Step {idx + 1}: Checking tool '{actual_tool}'...")

                # Check tool name selection
                if actual_tool != expected_step["expected_tool"]:
                    print(f"Step FAIL: Expected tool '{expected_step['expected_tool']}', got '{actual_tool}'")
                    scenario_passed = False
                    break

                # Run Deterministic Check on static arguments
                det_score = 1.0
                if expected_step["deterministic_args"]:
                    actual_det = {k: v for k, v in actual_args.items() if k in expected_step["deterministic_args"]}
                    det_result = await deterministic_evaluator(
                        outputs=actual_det,
                        reference_outputs=expected_step["deterministic_args"]
                    )
                    # Average the list of items directly if it's a raw list of scores/items
                    if isinstance(det_result, list) and len(det_result) > 0:
                        total_det = sum(
                            item.get("score", 0.0) if isinstance(item, dict) else float(item) 
                            for item in det_result
                        )
                        det_score = total_det / len(det_result)
                    elif isinstance(det_result, dict):
                        det_score = det_result.get("score", 0.0)
                    else:
                        det_score = 0.0

                # Run Semantic Check on fluid/relative arguments
                sem_score = 1.0
                if expected_step["semantic_args"]:
                    actual_sem = {k: v for k, v in actual_args.items() if k in expected_step["semantic_args"]}
                   
                    # Get the current UTC day
                    # today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d") # e.g., "2026-10-03"
            
                    # dynamic_references = {}
                    # for key, reference_val in expected_step["semantic_args"].items():
                    #     if "lower bound" in reference_val.lower() or "timemin" in key.lower():
                    #         dynamic_references[key] = f"{today_str}T00:00:00Z"
                    #     elif "upper bound" in reference_val.lower() or "timemax" in key.lower():
                    #         dynamic_references[key] = f"{today_str}T23:59:59Z"
                    #     else:
                    #         # Fallback or general query matching
                    #         dynamic_references[key] = reference_val  

                    # Run the test case
                    sem_result = await asyncio.to_thread(semantic_json_evaluator,
                        inputs=scenario['input'],
                        outputs=actual_sem,
                        reference_outputs=expected_step["semantic_args"]
                    )         

                    sem_score = sem_result['score']                    

                # Step Verification
                # Check if sem_score is explicitly a boolean True, or if it meets the numerical threshold
                sem_passed = sem_score is True or (isinstance(sem_score, (int, float)) and sem_score >= 0.9)
                det_passed = det_score >= 0.9

                if not det_passed or not sem_passed:
                    print(f"Step FAIL: Parameters did not match perfectly (Det: {det_score}, Sem: {sem_score})")
                    scenario_passed = False
                    break
                else:
                    scenario_passed = True
                    print(f"Step PASS")

        if scenario_passed:
            print(f"Overall Status: PASS")
        else:
            print(f"Overall Status: FAIL")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())
