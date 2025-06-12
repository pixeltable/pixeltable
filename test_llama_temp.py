import pixeltable as pxt
from pixeltable.functions.llama import chat_completions
import os
import traceback
import json # New import
import pixeltable.func as pxt_func # Import for UDF

# Actual UDF that will be called by the LLM via invoke_tools
@pxt_func.udf
def get_current_weather(location: str, unit: str = 'fahrenheit') -> str:
    """Get the current weather in a given location."""
    # In a real scenario, this would call a weather API
    # For this test, we'll return a fixed string based on location
    print(f"TOOL EXECUTED: get_current_weather(location='{location}', unit='{unit}')")
    if "san francisco" in location.lower():
        return json.dumps({"location": location, "temperature": "70", "unit": unit, "forecast": "sunny"})
    elif "tokyo" in location.lower():
        return json.dumps({"location": location, "temperature": "25", "unit": "celsius", "forecast": "cloudy"})
    else:
        return json.dumps({"location": location, "temperature": "unknown", "forecast": "unknown"})

# UDF for safely extracting message content (from previous step, still useful)
@pxt_func.udf
def safe_extract_content(message_dict: dict) -> str:
    if message_dict is None:
        return None
    return message_dict.get('content')

# UDF to format messages for the second LLM call (synthesis)
@pxt_func.udf
def format_tool_response_messages(initial_user_messages: list, llm_1_response: dict, tool_invocation_results: dict) -> list:
    """Formats the message list for the synthesis LLM call after tools have been invoked."""
    messages_for_synthesis = list(initial_user_messages) 

    if not llm_1_response or 'choices' not in llm_1_response or not llm_1_response['choices']:
        print("Warning (format_tool_response_messages): LLM1 response is invalid or empty.")
        return messages_for_synthesis

    assistant_message_dict = llm_1_response['choices'][0].get('message')
    if not assistant_message_dict:
        print("Warning (format_tool_response_messages): Assistant message missing in LLM1 response.")
        return messages_for_synthesis
        
    messages_for_synthesis.append(dict(assistant_message_dict)) # Append assistant's turn (with tool_calls)

    # Keep track of the next result index to use for each tool function,
    # in case a tool function is called multiple times by the LLM in one turn.
    tool_result_consumed_indices = {key: 0 for key in tool_invocation_results.keys()}

    requested_tool_calls = assistant_message_dict.get('tool_calls')
    if requested_tool_calls:
        for requested_tool_call in requested_tool_calls:
            tool_call_id = requested_tool_call.get('id')
            function_info = requested_tool_call.get('function')
            if not tool_call_id or not function_info:
                print(f"Warning (format_tool_response_messages): Invalid requested_tool_call structure: {requested_tool_call}")
                continue
            
            function_name = function_info.get('name')
            if not function_name:
                print(f"Warning (format_tool_response_messages): Missing function name in requested_tool_call: {requested_tool_call}")
                continue

            if function_name in tool_invocation_results:
                results_for_function = tool_invocation_results[function_name]
                current_idx = tool_result_consumed_indices.get(function_name, 0)

                if results_for_function is not None and current_idx < len(results_for_function):
                    tool_output_content = results_for_function[current_idx]
                    messages_for_synthesis.append({
                        'role': 'tool',
                        'tool_call_id': tool_call_id,
                        'content': str(tool_output_content) # Ensure content is a string
                    })
                    tool_result_consumed_indices[function_name] = current_idx + 1
                else:
                    err_msg = f"Error: Tool '{function_name}' (ID: {tool_call_id}) was called by LLM, but its output was not found, already used, or UDF returned None/empty."
                    print(f"Warning (format_tool_response_messages): {err_msg} Results for function: {results_for_function}, current_idx: {current_idx}")
                    messages_for_synthesis.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': err_msg})
            else:
                err_msg = f"Error: Tool '{function_name}' (ID: {tool_call_id}) was called by LLM, but no such tool name found in invoke_tools results."
                print(f"Warning (format_tool_response_messages): {err_msg} Available tool results for: {list(tool_invocation_results.keys())}")
                messages_for_synthesis.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': err_msg})
    
    # print(f"Debug (format_tool_response_messages): Messages for Synthesis LLM: {json.dumps(messages_for_synthesis, indent=2)}")
    return messages_for_synthesis

def main():
    print("Starting Llama integration test script...")
    try:
        # Initialize Pixeltable environment
        print("Initializing Pixeltable environment...")
        pxt.init() # Initialize the pixeltable environment
        # configure_logging can be called after init if needed, e.g.:
        # pxt.configure_logging(level='ERROR') 
        print("Pixeltable environment initialized.")

        # Define the model and a simple test prompt
        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'
        prompt_content = "What's the weather like in San Francisco?"
        table_name = 'temp_llama_test_table'

        print(f"Attempting to use Llama model: {test_model}")
        print(f"Prompt: {prompt_content}")

        # Clean up table if it exists from a previous failed run
        try:
            pxt.drop_table(table_name, if_not_exists='ignore') # Updated to use pxt.drop_table and if_not_exists
            print(f"Ensured table '{table_name}' does not exist before starting.")
        except Exception as e_drop_prev:
            # This might still catch broader issues, but direct drop_table errors should be handled by ignore_errors
            print(f"Note: Error during pre-run table cleanup (continuing): {e_drop_prev}")

        # Create a simple table
        print(f"Creating temporary table '{table_name}'...")
        # Define table schema
        schema = {
            'prompt_text_col': pxt.String, # Use the type class directly
            'initial_messages_col': pxt.Json # Use the type class directly
        }
        tbl = pxt.create_table(table_name, schema) # Updated to use pxt.create_table
        print(f"Table '{table_name}' created.")

        # Add data
        print(f"Inserting data into '{table_name}'...")
        messages_col_format = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt_content}
        ]
        tbl.insert([{'prompt_text_col': prompt_content, 'initial_messages_col': messages_col_format}]) # Wrap row in a list
        print("Inserted 1 row with 0 errors.")

        # Add a computed column for the Llama response
        print("Adding computed column for Llama response...")
        # Register UDFs as tools for the LLM
        # pxt.tools() will generate the schema from the UDF's signature and docstring
        llm_tools = pxt.tools(get_current_weather)
        # The llm_tools object is directly usable in chat_completions.
        # Inspecting its generated spec might require a different approach if needed for debugging.

        # 1. First LLM Call (Tool Selection)
        tbl.add_computed_column(llm_response_1=pxt.functions.llama.chat_completions(
            model=test_model,
            messages=tbl.initial_messages_col, # Use the column here
            tools=llm_tools # Pass the registered tools
            # Optional: Add other parameters like max_tokens, temperature, etc.
        ))
        print("Added llm_response_1 column (tool selection call).")

        # 2. Invoke Tools based on LLM 1's response
        # invoke_tools is imported from pixeltable.functions.llama (which re-exports from openai/anthropic)
        tbl.add_computed_column(tool_output=pxt.functions.llama.invoke_tools(llm_tools, tbl.llm_response_1))
        print("Added tool_output column.")

        # For inspection: extract the direct text response if any from LLM 1 (might be None if tool is called)
        tbl.add_computed_column(llm_1_direct_text=safe_extract_content(tbl.llm_response_1['choices'][0]['message']))
        print("Added llm_1_direct_text column for inspection.")

        # 3. Format messages for the second LLM call (Synthesis)
        tbl.add_computed_column(synthesis_messages=format_tool_response_messages(
            tbl.initial_messages_col, tbl.llm_response_1, tbl.tool_output
        ))
        print("Added synthesis_messages column.")

        # 4. Second LLM Call (Synthesis - no tools passed here, expect direct answer)
        tbl.add_computed_column(llm_synthesis_response=pxt.functions.llama.chat_completions(
            model=test_model,
            messages=tbl.synthesis_messages
            # No 'tools' argument here, we want a direct answer based on tool outputs
        ))
        print("Added llm_synthesis_response column.")

        # 5. Extract final answer from synthesis
        tbl.add_computed_column(final_answer_text=safe_extract_content(tbl.llm_synthesis_response['choices'][0]['message']))
        print("Added final_answer_text column.")

        # Collect and print results
        print("Collecting results...")
        result_df = tbl.collect()
        print("Results collected.")

        print("\n--- Llama API Response and Tool Usage --- ")
        if len(result_df) > 0:
            first_row = result_df[0]
            prompt_text_actual = first_row['prompt_text_col']
            llm_1_response_content = first_row['llm_response_1']
            actual_tool_output = first_row['tool_output']
            llm_1_text = first_row['llm_1_direct_text']
            synthesis_llm_messages = first_row['synthesis_messages'] # For inspection
            synthesis_response_content = first_row['llm_synthesis_response']
            final_answer = first_row['final_answer_text']

            print(f"Prompt: {prompt_text_actual}")
            print(f"LLM Response 1 (Tool Selection): {json.dumps(llm_1_response_content, indent=2)}")
            if llm_1_text:
                print(f"LLM 1 Direct Text Output: {llm_1_text}")
            else:
                print("LLM 1 Direct Text Output: None (tool call likely made)")
            
            print(f"Actual Tool Output (from invoke_tools): {actual_tool_output}")

            # Validation for this stage
            tool_calls_in_llm_1 = llm_1_response_content.get('choices', [{}])[0].get('message', {}).get('tool_calls')
            if tool_calls_in_llm_1:
                print("VALIDATION (LLM1): Tool call was requested by LLM1 - PASSED")
                
                # Check the output of invoke_tools for our specific UDF
                weather_tool_results_list = actual_tool_output.get('get_current_weather') # actual_tool_output is the dict from invoke_tools

                if weather_tool_results_list is not None and len(weather_tool_results_list) > 0:
                    tool_result_content = weather_tool_results_list[0] # Assuming one call to get_current_weather for this test
                    print(f"VALIDATION (invoke_tools): invoke_tools result for 'get_current_weather': [{tool_result_content}] - CHECKING")
                    
                    if tool_result_content is None: # This addresses the observed {'get_current_weather': [None]}
                        print("VALIDATION (UDF Output): 'get_current_weather' UDF returned None via invoke_tools - FAILED (Expected JSON string)")
                    else:
                        try:
                            # Ensure tool_result_content is a string before json.loads
                            tool_result_str = str(tool_result_content)
                            parsed_tool_result = json.loads(tool_result_str)
                            if "san francisco" in parsed_tool_result.get("location", "").lower() and parsed_tool_result.get("temperature") == "70":
                                print("VALIDATION (UDF Output): Tool UDF 'get_current_weather' provided correct JSON output for San Francisco - PASSED")
                            else:
                                print(f"VALIDATION (UDF Output): Tool UDF 'get_current_weather' output content unexpected: {tool_result_str} - FAILED")
                        except json.JSONDecodeError:
                            print(f"VALIDATION (UDF Output): Tool UDF 'get_current_weather' output was not valid JSON: {tool_result_str} - FAILED")
                        except Exception as e_parse: # Catch other potential errors during parsing (e.g., if not stringifiable)
                            print(f"VALIDATION (UDF Output): Error parsing tool UDF 'get_current_weather' output '{tool_result_content}': {e_parse} - FAILED")
                else:
                    print("VALIDATION (invoke_tools): 'get_current_weather' not found in invoke_tools output OR its result list was empty. Output: {actual_tool_output} - FAILED")
            else:
                print("VALIDATION (LLM1): No tool call requested by LLM1 for the weather prompt - FAILED")

            print(f"\nMessages Sent to Synthesis LLM: {json.dumps(synthesis_llm_messages, indent=2)}")
            print(f"LLM Synthesis Response: {json.dumps(synthesis_response_content, indent=2)}")
            print(f"Final Extracted Answer: {final_answer}")

            # Validation for final answer
            if final_answer and "70" in final_answer and ("sunny" in final_answer.lower() or "san francisco" in final_answer.lower()):
                print("VALIDATION (Final Answer): LLM provided a weather report for San Francisco using tool output - PASSED")
            else:
                print(f"VALIDATION (Final Answer): LLM did not provide expected weather report. Got: {final_answer} - FAILED")

        else:
            print("No response data collected.")
            print("Result DataFrame content (if any):")
            print(result_df)
        print("--- End of Llama API Response and Tool Usage ---")

    except Exception as e:
        print(f"\n--- ERROR during Llama integration test ---")
        print(f"An error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("--- End of ERROR ---   ")

    finally:
        # Clean up the temporary table
        # Check if 'tbl' was defined, implying table creation was attempted/successful
        if 'tbl' in locals() and tbl is not None:
            try:
                print(f"\nCleaning up temporary table '{table_name}'...")
                pxt.drop_table(table_name, if_not_exists='ignore') # Updated to use pxt.drop_table
                print(f"Table '{table_name}' cleaned up.")
            except Exception as e_cleanup:
                print(f"Error cleaning up table '{table_name}': {e_cleanup}")
        else:
            # If tbl was not created, still try to drop by name in case of partial setup
            try:
                print(f"\nAttempting cleanup of table '{table_name}' by name (if it exists)...")
                pxt.drop_table(table_name, if_not_exists='ignore')
                print(f"Table '{table_name}' (if it existed) cleaned up by name.")
            except Exception as e_cleanup_by_name:
                print(f"Error during final cleanup by name for table '{table_name}': {e_cleanup_by_name}")
        print("Llama integration test script finished.")

if __name__ == '__main__':
    main()
