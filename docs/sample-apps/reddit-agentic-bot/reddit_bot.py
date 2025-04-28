import praw
import prawcore # Import for exception handling
import pixeltable as pxt
import time
import os
from dotenv import load_dotenv
import signal
import sys
import threading # Import threading
from datetime import datetime, timezone

# Import centralized configuration
import config

# --- Configuration ---
# Force load_dotenv to override existing variables
load_dotenv(override=True)
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
USERNAME = os.getenv("REDDIT_USERNAME")
PASSWORD = os.getenv("REDDIT_PASSWORD")
TARGET_SUBREDDIT = config.TARGET_SUBREDDIT

if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT, USERNAME, PASSWORD]):
    print("Error: Missing Reddit API credentials in .env file")
    sys.exit(1)

# --- Globals ---
stop_event = threading.Event() # Use threading Event for signaling


# --- Signal Handler ---
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Signaling threads to stop...")
    stop_event.set() # Set the event to signal threads


signal.signal(signal.SIGINT, signal_handler)


# --- Helper Functions ---
def initialize_reddit():
    """Initializes and returns a PRAW Reddit instance."""
    print("Initializing Reddit client...")
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD,
        )

        reddit.user.me()  # Check authentication
        print(f"Successfully authenticated as Reddit user: {reddit.user.me()}")
        return reddit
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        sys.exit(1)


def initialize_pixeltable():
    """Connects to Pixeltable and gets table handles."""
    print("Connecting to Pixeltable...")
    try:
        questions_table_path = f"{config.BASE_DIR}.questions"
        questions_table = pxt.get_table(questions_table_path)
        print("Successfully connected to Pixeltable 'questions' table.")
        return questions_table
    except Exception as e:
        print(f"Error connecting to Pixeltable: {e}")
        print(
            f"Ensure the Pixeltable schema exists at '{questions_table_path}' (run setup_pixeltable.py)."
        )
        sys.exit(1)


def format_reply(answer: str) -> str:
    """Formats the final reply string.
    Assumes the 'answer' string already contains any necessary source citations.
    """
    reply = f"{answer}\n\n---\n\n"
    github_repo_url = "https://github.com/pixeltable/pixeltable"
    reply += f"^(This answer combines info from indexed documents and web search, powered by [Pixeltable]({github_repo_url}).)"
    return reply


def _update_status(table: pxt.Table, reddit_id: str, new_status: str):
    """Helper function to update the status of a question in Pixeltable."""
    print(f"        Attempting to update status for {reddit_id} to '{new_status}'...")
    try:
        table.update(
            {"status": new_status},
            where=table.reddit_id == reddit_id,
        )
        print(f"        Successfully updated status for {reddit_id} to '{new_status}'.")
    except Exception as update_err:
        print(f"        ERROR updating status for {reddit_id} to '{new_status}': {update_err}")


# --- Main Logic ---

def listen_for_submissions(reddit, questions_table, bot_username):
    """Listens for new submissions and inserts valid ones into Pixeltable."""
    subreddit = reddit.subreddit(TARGET_SUBREDDIT)
    print(f"--- [Listener Thread] Started listening for submissions in r/{subreddit.display_name} ---")
    try:
        for submission in subreddit.stream.submissions(skip_existing=True):
            if stop_event.is_set():
                print("[Listener Thread] Stop event detected, exiting listener loop.")
                break

            if submission is None: # Check if stream returned None (can happen)
                time.sleep(1) # Avoid busy-waiting if stream pauses
                continue

            reddit_id = submission.id
            sub_author_obj = submission.author
            sub_author = sub_author_obj.name if sub_author_obj else "[deleted]"
            sub_is_self = submission.is_self

            print(f"[Listener Thread] Seen submission: {reddit_id}")

            if not sub_is_self:
                print(f"[Listener Thread]  -> Skipping {reddit_id}: Not a self-post.")
                continue
            if sub_author == bot_username:
                print(f"[Listener Thread]  -> Skipping {reddit_id}: Authored by bot.")
                continue

            print(f"[Listener Thread] --> Processing valid new submission: {reddit_id}")
            question_text = f"Title: {submission.title}\n\nBody: {submission.selftext}"
            timestamp = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            insert_data = {
                "reddit_id": reddit_id,
                "subreddit": subreddit.display_name,
                "author": sub_author,
                "question_text": question_text,
                "timestamp": timestamp,
                "status": "new",
            }

            try:
                print(f"[Listener Thread]    Inserting {reddit_id} into Pixeltable...")
                status = questions_table.insert([insert_data])
                exceptions_list = getattr(status, 'exceptions', None)
                num_exceptions = len(exceptions_list) if exceptions_list else 0
                if status.num_rows == 1 and num_exceptions == 0:
                     print(f"[Listener Thread]    Insertion successful for {reddit_id}.")
                else:
                     print(f"[Listener Thread]    Insertion issue for {reddit_id}: {status.num_rows} inserted (expected 1), {num_exceptions} errors: {exceptions_list}")
            except Exception as e:
                print(f"[Listener Thread]    ERROR inserting {reddit_id} into Pixeltable: {e}")

            time.sleep(1)

    except prawcore.exceptions.RequestException as e:
        print(f"[Listener Thread] PRAW Request Error in stream: {e}. Listener stopping.")
    except prawcore.exceptions.ResponseException as e:
         print(f"[Listener Thread] PRAW Response Error in stream: {e}. Listener stopping.")
    except Exception as e:
        print(f"[Listener Thread] Unexpected error in submission stream: {e}. Listener stopping.")
    finally:
        print("--- [Listener Thread] Exiting --- ")

def check_and_reply(reddit, questions_table, bot_username):
    """Periodically checks Pixeltable for answered questions and replies to all available.

    Now processes all ready questions found in one cycle, instead of just the first.
    Uses a helper function for status updates.
    """
    print("  -> Entering check_and_reply function...")
    try:
        print("    Querying Pixeltable for questions to reply to...")
        answered_questions_results = (
            questions_table.where(
                (questions_table.status != "replied")
                & (questions_table.status != "error")
                & (questions_table.status != "deleted")
                & (questions_table.status != "reply_error")
                & (questions_table.final_answer != None)
            )
            .select(
                questions_table.reddit_id,
                questions_table.final_answer
            )
            .order_by(questions_table.timestamp)
            .collect()
        )

        if not answered_questions_results:
            print(
                "    No questions found ready to be replied to."
            )
            return

        print(
            f"    Found {len(answered_questions_results)} potential questions to reply to. Processing all..."
        )
        for question_data in answered_questions_results:
            if not isinstance(question_data, dict):
                print(f"Warning: Unexpected item type in results: {type(question_data)}. Skipping.")
                continue

            reddit_id = question_data.get("reddit_id")
            final_answer = question_data.get("final_answer")

            if not reddit_id:
                print("      Skipping row: Missing reddit_id.")
                continue

            print(f"      Processing potential reply for {reddit_id}...")

            if not final_answer:
                print(f"      Skipping {reddit_id}: Final answer is empty/null in Pixeltable.")
                _update_status(questions_table, reddit_id, "error")
                continue

            try:
                print(f"        Fetching submission {reddit_id} to check existing comments...")
                submission = reddit.submission(id=reddit_id)
                if not submission:
                     print(f"        Submission {reddit_id} not found or deleted.")
                     _update_status(questions_table, reddit_id, "deleted")
                     continue

                already_replied = False
                print(f"        Checking comments for {reddit_id} for bot user '{bot_username}'...")
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if comment.author and comment.author.name == bot_username:
                        already_replied = True
                        print(f"        Bot '{bot_username}' has already commented on {reddit_id}. Skipping.")
                        break

                if already_replied:
                    _update_status(questions_table, reddit_id, "replied")
                    continue

                print(f"        Bot has not replied to {reddit_id} yet. Proceeding with reply.")
                reply_body = format_reply(final_answer)

                try:
                    print(f"        Posting reply to submission {reddit_id}...")
                    submission.reply(reply_body)
                    print(f"        Successfully replied to {reddit_id}.")
                    _update_status(questions_table, reddit_id, "replied")

                except praw.exceptions.APIException as e:
                    print(f"        PRAW API Error replying to {reddit_id}: {e}")
                    if "RATELIMIT" in str(e):
                        print("        Rate limit hit. Will retry next cycle.")
                    else:
                        _update_status(questions_table, reddit_id, "reply_error")

                except Exception as e:
                    print(f"        Unexpected error during reply/update for {reddit_id}: {e}")
                    _update_status(questions_table, reddit_id, "reply_error")

            except Exception as outer_err:
                print(f"      ERROR checking submission/comments for {reddit_id}: {outer_err}")
                _update_status(questions_table, reddit_id, "error")
                continue

            print(f"      --- Pausing briefly after processing {reddit_id} ---")
            time.sleep(5)

    except Exception as e:
        print(f"  ERROR during check_and_reply function query/iteration: {e}")

    print("  -> Exiting check_and_reply function.")


# --- Main Execution (Threading) ---
if __name__ == "__main__":
    print("Starting Reddit Bot (Listener/Reply Mode)...")
    reddit_client = initialize_reddit()
    pxt_questions_table = initialize_pixeltable()

    bot_username = None
    try:
        bot_username = reddit_client.user.me().name
        print(f"Bot username confirmed: {bot_username}")
    except Exception as e:
        print(f"Error getting bot username: {e}")
        sys.exit(1)

    print("Starting listener thread...")
    listener_thread = threading.Thread(
        target=listen_for_submissions,
        args=(reddit_client, pxt_questions_table, bot_username),
        daemon=True
    )
    listener_thread.start()
    print("Listener thread started.")

    print(
        f"Entering main reply check loop (checking every {config.CHECK_INTERVAL} seconds). Press Ctrl+C to stop."
    )
    loop_count = 0
    try:
        while not stop_event.is_set():
            loop_count += 1
            print(f"\n--- Running Reply Check Cycle #{loop_count} --- ")
            try:
                print(f"Cycle #{loop_count}: Calling check_and_reply...")
                check_and_reply(reddit_client, pxt_questions_table, bot_username)
                print(f"Cycle #{loop_count}: Finished check_and_reply.")

            except Exception as e:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR in main check_and_reply loop (Cycle #{loop_count}): {e}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                time.sleep(min(config.CHECK_INTERVAL, 15))

            print(
                f"--- Reply Check Cycle #{loop_count} Complete. Sleeping for {config.CHECK_INTERVAL} seconds... ---"
            )
            stop_event.wait(config.CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main loop. Stopping.")
        stop_event.set()

    finally:
        print("\nMain loop finished. Waiting for listener thread to exit (if needed)...")
        print("Reddit Bot finished.")
        sys.exit(0)
