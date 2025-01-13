# main.py
import logging
import argparse
import pixeltable as pxt
from config import check_api_key
from research_agent import create_research_table

# Configure loggers
logging.getLogger('pixeltable_pgserver').setLevel(logging.WARNING)

# Configure root logger for your app
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def process_query(query: str) -> None:
    logger.info(f"Processing query: {query}")

    try:
        research_table = pxt.get_table('research.queries')
    except Exception:
        research_table = create_research_table()

    research_table.insert([{'input': query}])

    # Get results - using initial_answer instead of final_answer
    result = research_table.select(
        research_table.initial_answer,  # Changed from final_answer
        research_table.final_summary
    ).tail(1)

    # Display results
    logger.info("\nAnalysis:")
    logger.info(result['initial_answer'][0])  # Changed from final_answer
    logger.info("\nSummary:")
    logger.info(result['final_summary'][0])

def main() -> None:
    """Main execution flow with command line arguments."""
    parser = argparse.ArgumentParser(description='Research Assistant CLI')
    parser.add_argument('--query', '-q', type=str, help='Custom research query')
    parser.add_argument('--sample', '-s', action='store_true',
                       help='Run sample queries')
    args = parser.parse_args()

    try:
        check_api_key('OPENAI_API_KEY')
        check_api_key('NEWS_API_KEY')

        if args.query:
            process_query(args.query)

        if args.sample or not args.query:
            SAMPLE_QUERIES = [
                "What can you tell me about Nvidia?",
                "Summarize key insights from the Zacks report"
            ]
            for query in SAMPLE_QUERIES:
                process_query(query)
                logger.info("="*50)

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()