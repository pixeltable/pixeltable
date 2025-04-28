import pixeltable as pxt
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
import yfinance as yf


@pxt.udf()
def run_duckduckgo_search(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Performs a DuckDuckGo search and returns results as a list of dicts."""
    # print(f"Performing DuckDuckGo search for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        # print(f"  Found {len(results)} results.")
        return results if results else None
    except Exception:
        # print(f"Error during DuckDuckGo search: {e}")
        return None

# Tool UDF: Fetches financial data using yfinance.
# Integrates external Python libraries into the Pixeltable workflow.
# Registered as a tool for the LLM via `pxt.tools()` in setup_pixeltable.py.
@pxt.udf
def fetch_financial_data(ticker: str) -> str:
    """Fetch financial summary data for a given company ticker using yfinance."""
    try:
        if not ticker:
            return "Error: No ticker symbol provided."

        stock = yf.Ticker(ticker)

        # Get the info dictionary - this is the primary source now
        info = stock.info
        if (
            not info or info.get("quoteType") == "MUTUALFUND"
        ):  # Basic check if info exists and isn't a mutual fund (less relevant fields)
            # Attempt history for basic validation if info is sparse
            hist = stock.history(period="1d")
            if hist.empty:
                return f"Error: No data found for ticker '{ticker}'. It might be delisted or incorrect."
            else:  # Sometimes info is missing but history works, provide minimal info
                return f"Limited info for '{ticker}'. Previous Close: {hist['Close'].iloc[-1]:.2f} (if available)."

        # Select and format key fields from the info dictionary
        data_points = {
            "Company Name": info.get("shortName") or info.get("longName"),
            "Symbol": info.get("symbol"),
            "Exchange": info.get("exchange"),
            "Quote Type": info.get("quoteType"),
            "Currency": info.get("currency"),
            "Current Price": info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("bid"),
            "Previous Close": info.get("previousClose"),
            "Open": info.get("open"),
            "Day Low": info.get("dayLow"),
            "Day High": info.get("dayHigh"),
            "Volume": info.get("volume") or info.get("regularMarketVolume"),
            "Market Cap": info.get("marketCap"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "Dividend Yield": info.get("dividendYield"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "Avg Volume (10 day)": info.get("averageDailyVolume10Day"),
            # Add more fields if desired
        }

        formatted_data = [
            f"Financial Summary for {data_points.get('Company Name', ticker)} ({data_points.get('Symbol', ticker).upper()}) - {data_points.get('Quote Type', 'N/A')}"
        ]
        formatted_data.append("-" * 40)

        for key, value in data_points.items():
            if value is not None:  # Only show fields that have a value
                formatted_value = value
                # Format specific types for readability
                if key in [
                    "Current Price",
                    "Previous Close",
                    "Open",
                    "Day Low",
                    "Day High",
                    "52 Week Low",
                    "52 Week High",
                ] and isinstance(value, (int, float)):
                    formatted_value = (
                        f"{value:.2f} {data_points.get('Currency', '')}".strip()
                    )
                elif key in [
                    "Volume",
                    "Market Cap",
                    "Avg Volume (10 day)",
                ] and isinstance(value, (int, float)):
                    if value > 1_000_000_000:
                        formatted_value = f"{value / 1_000_000_000:.2f}B"
                    elif value > 1_000_000:
                        formatted_value = f"{value / 1_000_000:.2f}M"
                    elif value > 1_000:
                        formatted_value = f"{value / 1_000:.2f}K"
                    else:
                        formatted_value = f"{value:,}"
                elif key == "Dividend Yield" and isinstance(value, (int, float)):
                    formatted_value = f"{value * 100:.2f}%"
                elif (
                    key == "Trailing P/E"
                    or key == "Forward P/E") and isinstance(value, (int, float)
                ):
                    formatted_value = f"{value:.2f}"

                formatted_data.append(f"{key}: {formatted_value}")

        # Optionally, add a line about latest financials if easily available
        try:
            latest_financials = stock.financials.iloc[:, 0]
            revenue = latest_financials.get("Total Revenue")
            net_income = latest_financials.get("Net Income")
            if revenue is not None or net_income is not None:
                formatted_data.append("-" * 40)
                fin_date = latest_financials.name.strftime("%Y-%m-%d")
                if revenue:
                    formatted_data.append(
                        f"Latest Revenue ({fin_date}): ${revenue / 1e6:.2f}M"
                    )
                if net_income:
                    formatted_data.append(
                        f"Latest Net Income ({fin_date}): ${net_income / 1e6:.2f}M"
                    )
        except Exception:
            pass  # Ignore errors fetching/parsing financials for this summary

        return "\n".join(formatted_data)

    except Exception as e:
        # traceback.print_exc()  # REMOVED: Log the full error for debugging
        return f"Error fetching financial data for {ticker}: {str(e)}."


@pxt.udf()
def format_initial_prompt(
    question_text: str, retrieved_context: Optional[List[Dict]]
) -> List[Dict]:
    """Formats the *initial* prompt message list for the LLM.
       Uses related_url for source display if available.
    """
    context_str = "No relevant documents found.\n"
    if retrieved_context:
        context_items = []
        # Use a dict to track seen sources and their preferred display name
        sources_seen = {}
        for item in retrieved_context:
            if item and "text" in item:
                text = item["text"]
                source_uri = item.get("source_uri")
                related_url = item.get("related_url")

                # Determine the primary identifier and display name for this source
                # Prefer related_url if it exists, otherwise use source_uri
                primary_id = related_url if related_url else source_uri
                display_source = primary_id if primary_id else "Unknown Source"

                if not primary_id:
                    continue # Skip if we have neither identifier

                if primary_id not in sources_seen:
                    # First time seeing this source (by its primary ID)
                    sources_seen[primary_id] = display_source # Store its display name
                    # Optionally add original path if related_url was used
                    source_prefix = f"Source: {display_source}"
                    if related_url and source_uri:
                         source_prefix += f" (Original: {source_uri})"
                    context_items.append(f"{source_prefix}\nContent: {text}")
                else:
                    # Source already seen, just append content
                    context_items.append(f"Content: {text}")

        if context_items:
            context_str = "\n\n".join(context_items)
        else:
            context_str = "No processable context items found.\n"

    messages = [
        {"role": "user", "content": f"""Context Documents:\n---\n{context_str}\n---\n\nQuestion: {question_text}"""}
    ]
    return messages


@pxt.udf()
def format_synthesis_messages(
    question_text: str,
    retrieved_context: Optional[List[Dict]],
    tool_output: Optional[Dict],
    llm_general_response: Optional[Dict],
) -> List[Dict]:
    """Formats the user message for the final synthesis LLM call.
       Uses related_url for source display if available.
    """
    # 1. Format Document Context
    doc_context_str = "No relevant documents found."
    if retrieved_context:
        context_items = []
        # Use a dict to track seen sources and their preferred display name
        sources_seen_synthesis = {}
        for item in retrieved_context:
            if item and "text" in item:
                text = item["text"]
                source_uri = item.get("source_uri")
                related_url = item.get("related_url")

                # Determine the primary identifier and display name for this source
                # Prefer related_url if it exists, otherwise use source_uri
                primary_id = related_url if related_url else source_uri
                display_source = primary_id if primary_id else "Unknown Source"

                if not primary_id:
                     continue # Skip if we have neither identifier

                if primary_id not in sources_seen_synthesis:
                    # First time seeing this source (by its primary ID)
                    sources_seen_synthesis[primary_id] = display_source
                    # Optionally add original path if related_url was used
                    source_prefix = f"Source: {display_source}"
                    if related_url and source_uri:
                         source_prefix += f" (Original: {source_uri})"
                    context_items.append(f"{source_prefix}\nContent: {text}")
                else:
                    # Source already seen, just append content
                    context_items.append(f"Content: {text}")

        if context_items:
            doc_context_str = "\n\n".join(context_items)
        else:
            # Added else case for clarity if loop runs but finds no processable items
            doc_context_str = "No processable context items found."

    # 2. Format Web Search Results (String conversion)
    tool_output_str = "No web search performed or failed."
    if tool_output and isinstance(tool_output, dict) and tool_output:
        tool_output_str = f"Web Search Results Dictionary:\n{str(tool_output)}"

    # 3. Format General Knowledge Answer
    general_answer_str = "No general knowledge answer was generated."
    if llm_general_response and isinstance(llm_general_response, dict):
        content_list = llm_general_response.get("content")
        if isinstance(content_list, list) and content_list:
            first_block = content_list[0]
            if isinstance(first_block, dict) and first_block.get("type") == "text":
                general_answer_str = first_block.get("text", general_answer_str)

    # 4. Construct the final prompt message content
    final_content = f"""Original Question: {question_text}

--- Source 1: Document Context ---
{doc_context_str}
--- End Document Context ---

--- Source 2: Web Search Raw Output ---
{tool_output_str}
--- End Web Search Raw Output ---

--- Source 3: General Knowledge Answer ---
{general_answer_str}
--- End General Knowledge Answer ---

Please synthesize the final answer based on the instructions provided in the system message, using the sources above.
"""
    messages = [
        {"role": "user", "content": final_content.strip()}
    ]
    return messages
