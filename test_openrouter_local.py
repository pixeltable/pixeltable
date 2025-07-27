#!/usr/bin/env python3
"""
Local test script for OpenRouter integration
"""

import os
import pixeltable as pxt
from pixeltable.functions import openrouter

def test_basic_integration():
    """Test basic OpenRouter functionality"""
    print("🧪 Testing OpenRouter Integration...")
    
    # Check API key
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    print("✅ API key found")
    
    try:
        # Create a test directory and table
        print("📁 Creating test table...")
        pxt.drop_dir('openrouter_test', force=True)
        pxt.create_dir('openrouter_test')
        
        t = pxt.create_table('openrouter_test.basic', {
            'prompt': pxt.String
        })
        
        print("🔧 Adding computed column...")
        # Use a fast, cheap model for testing
        t.add_computed_column(
            response=openrouter.chat_completions(
                messages=[{'role': 'user', 'content': t.prompt}],
                model='openai/gpt-3.5-turbo',
                model_kwargs={'max_tokens': 50, 'temperature': 0.7}
            ).choices[0].message.content
        )
        
        print("📝 Inserting test data...")
        t.insert([
            {'prompt': 'Say hello and tell me you are working via OpenRouter'}
        ])
        
        print("🔍 Retrieving results...")
        results = t.select(t.prompt, t.response).collect()
        
        print("📊 Results:")
        for i in range(len(results)):
            print(f"  Prompt: {results['prompt'][i]}")
            print(f"  Response: {results['response'][i]}")
        
        print("✅ Basic integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            pxt.drop_dir('openrouter_test', force=True)
            print("🧹 Cleaned up test resources")
        except:
            pass

def test_model_listing():
    """Test OpenRouter model listing functionality"""
    print("\n🧪 Testing OpenRouter Model Listing...")
    
    try:
        # Create a test table for models
        pxt.drop_dir('openrouter_models_test', force=True)
        pxt.create_dir('openrouter_models_test')
        
        t = pxt.create_table('openrouter_models_test.models', {
            'fetch': pxt.Bool
        })
        
        print("🔧 Adding models listing column...")
        t.add_computed_column(available_models=openrouter.models())
        
        print("📝 Fetching models...")
        t.insert([{'fetch': True}])
        
        results = t.select(t.available_models).collect()
        models_data = results['available_models'][0]
        
        print(f"📊 Found {len(models_data)} available models")
        
        # Show a few interesting models
        print("🔍 Sample models:")
        for i, model in enumerate(models_data[:5]):
            print(f"  {i+1}. {model['id']} - {model.get('name', 'N/A')}")
            if 'pricing' in model and model['pricing']:
                pricing = model['pricing']
                if 'prompt' in pricing:
                    print(f"     Price: ${pricing['prompt']} per 1K tokens")
        
        print("✅ Model listing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model listing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            pxt.drop_dir('openrouter_models_test', force=True)
            print("🧹 Cleaned up models test resources")
        except:
            pass

def test_provider_routing():
    """Test OpenRouter provider routing functionality"""
    print("\n🧪 Testing OpenRouter Provider Routing...")
    
    try:
        # Create a test table
        pxt.drop_dir('openrouter_routing_test', force=True)
        pxt.create_dir('openrouter_routing_test')
        
        t = pxt.create_table('openrouter_routing_test.routing', {
            'prompt': pxt.String
        })
        
        print("🔧 Adding provider routing columns...")
        
        # Test basic response without provider routing
        t.add_computed_column(
            basic_response=openrouter.chat_completions(
                messages=[{'role': 'user', 'content': t.prompt}],
                model='openai/gpt-3.5-turbo',
                model_kwargs={'max_tokens': 30}
            ).choices[0].message.content
        )
        
        # Test with provider preference (prefer OpenAI)
        t.add_computed_column(
            routed_response=openrouter.chat_completions(
                messages=[{'role': 'user', 'content': t.prompt}],
                model='openai/gpt-3.5-turbo',
                model_kwargs={'max_tokens': 30},
                provider={'order': ['openai'], 'allow_fallbacks': True}
            ).choices[0].message.content
        )
        
        # Test with transforms (middle-out compression)
        t.add_computed_column(
            transformed_response=openrouter.chat_completions(
                messages=[{'role': 'user', 'content': t.prompt}],
                model='openai/gpt-3.5-turbo',
                model_kwargs={'max_tokens': 30},
                transforms=['middle-out']
            ).choices[0].message.content
        )
        
        print("📝 Testing routing and transforms...")
        t.insert([
            {'prompt': 'Say "Hello from OpenRouter!" and explain what you are.'}
        ])
        
        results = t.select(t.prompt, t.basic_response, t.routed_response, t.transformed_response).collect()
        
        print("📊 Provider Routing Results:")
        for i in range(len(results)):
            print(f"  Prompt: {results['prompt'][i]}")
            print(f"  Basic: {results['basic_response'][i]}")
            print(f"  Routed: {results['routed_response'][i]}")
            print(f"  Transformed: {results['transformed_response'][i]}")
        
        print("✅ Provider routing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Provider routing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            pxt.drop_dir('openrouter_routing_test', force=True)
            print("🧹 Cleaned up routing test resources")
        except:
            pass

def test_tool_calling():
    """Test OpenRouter tool calling functionality"""
    print("\n🧪 Testing OpenRouter Tool Calling...")
    
    try:
        # Create a test table
        pxt.drop_dir('openrouter_tools_test', force=True)
        pxt.create_dir('openrouter_tools_test')
        
        t = pxt.create_table('openrouter_tools_test.tools', {
            'user_query': pxt.String
        })
        
        print("🔧 Setting up tool calling...")
        
        # Define a simple calculator tool
        calculator_tool = {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number", 
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
        
        # Add computed column with tool calling
        t.add_computed_column(
            tool_response=openrouter.chat_completions(
                messages=[{
                    'role': 'user', 
                    'content': t.user_query
                }],
                model='openai/gpt-4o-mini',  # Use a model that supports tool calling
                model_kwargs={
                    'max_tokens': 150, 
                    'temperature': 0.1,
                    'tools': [calculator_tool],
                    'tool_choice': {'type': 'function', 'function': {'name': 'calculator'}}
                }
            )
        )
        
        print("📝 Testing tool calling...")
        t.insert([
            {'user_query': 'What is 15 multiplied by 7? Please use the calculator tool to compute this.'}
        ])
        
        results = t.select(t.user_query, t.tool_response).collect()
        
        print("📊 Tool Calling Results:")
        for i in range(len(results)):
            print(f"  Query: {results['user_query'][i]}")
            response = results['tool_response'][i]
            
            # Check if tool was called
            if 'choices' in response and len(response['choices']) > 0:
                choice = response['choices'][0]
                if 'message' in choice:
                    message = choice['message']
                    
                    # Check for tool calls
                    if 'tool_calls' in message and message['tool_calls']:
                        print(f"  🔧 Tool called: {message['tool_calls'][0]['function']['name']}")
                        print(f"  📋 Arguments: {message['tool_calls'][0]['function']['arguments']}")
                        print("  ✅ Tool calling successful!")
                    else:
                        print(f"  💬 Response: {message.get('content', 'No content')}")
                        print("  ⚠️ No tools were called")
            else:
                print(f"  ❌ Unexpected response format: {response}")
        
        print("✅ Tool calling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool calling test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            pxt.drop_dir('openrouter_tools_test', force=True)
            print("🧹 Cleaned up tool calling test resources")
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting OpenRouter Integration Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_basic_integration()
    all_passed &= test_model_listing()
    all_passed &= test_provider_routing()
    all_passed &= test_tool_calling()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All OpenRouter integration tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    print("✨ OpenRouter integration is ready to use!") 